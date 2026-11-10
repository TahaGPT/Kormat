"""Decoding API: no-cache reference, own KV-cache decoder, own greedy speculative decoder, HF generate reference.

Shapes (batch 1): input_ids (1,P). Every model call passes explicit ``attention_mask`` (1, total_kv_len),
``position_ids`` (1,q_len), ``cache_position`` (q_len,) so nothing is inferred implicitly.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

import psutil
import torch

from .cache import PreallocKVCache


@dataclass
class DecodeResult:
    tokens: list[int]
    text: str | None
    prefill_ms: float
    decode_ms: float
    n_prompt: int
    n_new: int
    tokens_per_s: float
    peak_rss_mb: float
    kv_cache_mb: float | None
    stats: dict = field(default_factory=dict)
    rss_start_mb: float = 0.0   # RSS at call entry (so weights baseline can be subtracted)


class _RSSPeak:
    """Samples process RSS in a thread (every 5 ms) -> high-water mark during the call."""

    def __enter__(self):
        self.p = psutil.Process()
        self.start = self.peak = self.p.memory_info().rss
        self._stop = threading.Event()
        self.t = threading.Thread(target=self._run, daemon=True)
        self.t.start()
        return self

    def _run(self):
        while not self._stop.is_set():
            self.peak = max(self.peak, self.p.memory_info().rss)
            time.sleep(0.005)

    def __exit__(self, *a):
        self._stop.set()
        self.t.join()
        self.peak = max(self.peak, self.p.memory_info().rss)

    @property
    def peak_mb(self):
        return self.peak / 2**20

    @property
    def start_mb(self):
        return self.start / 2**20


def kv_bytes(model, seq_len: int, batch: int = 1) -> int:
    """Analytical KV size: 2 (K and V) * layers * kv_heads * head_dim * seq_len * bytes/elt * batch."""
    c = model.config
    hd = c.hidden_size // c.num_attention_heads
    isz = next(model.parameters()).element_size()
    return 2 * c.num_hidden_layers * c.num_key_value_heads * hd * seq_len * isz * batch


def _finish(tokens, tok, t_pre, t_dec, P, rss, kv_mb, stats=None) -> DecodeResult:
    tot = t_pre + t_dec
    return DecodeResult(
        tokens=list(tokens), text=tok.decode(tokens, skip_special_tokens=True) if tok is not None else None,
        prefill_ms=t_pre * 1e3, decode_ms=t_dec * 1e3, n_prompt=P, n_new=len(tokens),
        tokens_per_s=len(tokens) / tot if tot > 0 else 0.0, peak_rss_mb=rss.peak_mb, kv_cache_mb=kv_mb,
        stats=stats or {}, rss_start_mb=rss.start_mb)


def _argmax(logits_row: torch.Tensor) -> int:
    return int(torch.argmax(logits_row, dim=-1))


# ----------------------------------------------------------------------------------------------------------
# Reference: no cache. Every step re-runs the WHOLE sequence (O(n^2) total attention/MLP work).
# ----------------------------------------------------------------------------------------------------------
@torch.inference_mode()
def greedy_nocache(model, input_ids, max_new_tokens, eos_ids, tokenizer=None, force_len=False, trace=None):
    ids = input_ids
    P = ids.shape[1]
    out: list[int] = []
    t_pre = t_dec = 0.0
    with _RSSPeak() as rss:
        for step in range(max_new_tokens):
            t0 = time.perf_counter()
            # use_cache=False: no past_key_values; position_ids default to arange(len); we only need the last row.
            lg = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False,
                       num_logits_to_keep=1).logits[0, -1]
            nxt = _argmax(lg)
            dt = time.perf_counter() - t0
            if step == 0:
                t_pre += dt      # first step == prefill (same work as the cached prefill)
            else:
                t_dec += dt
            if trace is not None:
                trace.append(lg.float().clone())
            out.append(nxt)
            if nxt in eos_ids and not force_len:
                break
            ids = torch.cat([ids, torch.tensor([[nxt]], dtype=ids.dtype)], dim=1)
    return _finish(out, tokenizer, t_pre, t_dec, P, rss, None)


# ----------------------------------------------------------------------------------------------------------
# Own KV-cache decoder.
# ----------------------------------------------------------------------------------------------------------
def _fwd(model, tokens: torch.Tensor, cache: PreallocKVCache, last_only=True):
    """Feed ``tokens`` (1,q) at positions [L, L+q) where L = cache.length. Returns logits (q or 1, V).

    position_ids = [[L..L+q-1]] -> RoPE phase of each new token (K is rotated BEFORE caching, so old keys keep the
    phase of their absolute position and never need re-rotation).
    cache_position = same indices; it is what the model uses to build the causal mask (key j visible to query i iff
    j <= cache_position[i]) and what our cache asserts against.
    attention_mask = ones(1, L+q): all L cached + q new keys are real (no padding at batch 1); its length defines the
    key axis of the 4-D mask.
    """
    L, q = cache.length, tokens.shape[1]
    pos = torch.arange(L, L + q, dtype=torch.long)
    out = model(input_ids=tokens, attention_mask=torch.ones(1, L + q, dtype=torch.long), position_ids=pos[None],
                past_key_values=cache, use_cache=True, cache_position=pos,
                num_logits_to_keep=1 if last_only else 0)
    assert out.past_key_values is cache and cache.length == L + q
    return out.logits[0]


@torch.inference_mode()
def greedy_kv(model, input_ids, max_new_tokens, eos_ids, tokenizer=None, force_len=False, trace=None):
    P = input_ids.shape[1]
    cache = PreallocKVCache.for_model(model, 1, P + max_new_tokens)
    out: list[int] = []
    with _RSSPeak() as rss:
        t0 = time.perf_counter()
        lg = _fwd(model, input_ids, cache)[-1]          # prefill: q_len = P, mask None -> is_causal path
        nxt = _argmax(lg)
        t_pre = time.perf_counter() - t0
        if trace is not None:
            trace.append(lg.float().clone())
        out.append(nxt)
        t0 = time.perf_counter()
        while len(out) < max_new_tokens and not (nxt in eos_ids and not force_len):
            lg = _fwd(model, torch.tensor([[nxt]]), cache)[-1]   # decode: q_len = 1 at position P+len(out)-1
            nxt = _argmax(lg)
            if trace is not None:
                trace.append(lg.float().clone())
            out.append(nxt)
        t_dec = time.perf_counter() - t0
    assert cache.update_calls > 0
    return _finish(out, tokenizer, t_pre, t_dec, P, rss, kv_bytes(model, cache.length) / 2**20,
                   {"cache_update_calls": cache.update_calls})


# ----------------------------------------------------------------------------------------------------------
# Greedy speculative decoding.
# ----------------------------------------------------------------------------------------------------------
@torch.inference_mode()
def speculative_greedy(target, draft, input_ids, max_new_tokens, eos_ids, gamma=4, tokenizer=None, force_len=False):
    """Invariant (asserted every iteration), with seq = prompt + emitted so far and ``pending`` = seq[-1]:
         target_cache.length == len(seq) - 1      (everything except the not-yet-fed pending token)
         draft_cache.length  <= len(seq) - 1      (draft may lag by the one token d_gamma it never consumed)
    """
    P = input_ids.shape[1]
    cap = P + max_new_tokens + gamma + 2
    tc = PreallocKVCache.for_model(target, 1, cap)
    dc = PreallocKVCache.for_model(draft, 1, cap)
    seq: list[int] = input_ids[0].tolist()
    out: list[int] = []
    per_iter = []
    drafted = accepted = 0

    def done():
        return len(out) >= max_new_tokens or (not force_len and out and out[-1] in eos_ids)

    with _RSSPeak() as rss:
        t0 = time.perf_counter()
        nxt = _argmax(_fwd(target, input_ids, tc)[-1])
        _fwd(draft, input_ids, dc)
        t_pre = time.perf_counter() - t0
        seq.append(nxt)
        out.append(nxt)
        t0 = time.perf_counter()
        while not done():
            n0 = len(seq)
            g = min(gamma, max_new_tokens - len(out) - 1)   # never draft past what we may still emit
            assert tc.length == n0 - 1 and dc.length <= n0 - 1
            drafts: list[int] = []
            # --- draft g tokens: first feed what the draft lacks (seq[dc.length:], 1 token, or 2 if last round a==gamma)
            if g > 0:
                feed = torch.tensor([seq[dc.length:]])
                for _ in range(g):
                    d = _argmax(_fwd(draft, feed, dc)[-1])
                    drafts.append(d)
                    feed = torch.tensor([[d]])
                # draft cache now holds seq[:n0-1] + pending + d1..d_{g-1} ; it has NOT consumed d_g
            # --- verify: one target forward on [pending, d1..dg]; row i predicts the token after position n0-1+i
            block = torch.tensor([[seq[-1]] + drafts])
            tgt = torch.argmax(_fwd(target, block, tc, last_only=False), dim=-1).tolist()   # length g+1
            a = 0
            while a < g and drafts[a] == tgt[a]:
                a += 1
            new = drafts[:a] + [tgt[a]]       # accepted drafts + the target's own token (correction or bonus)
            drafted += g
            accepted += a
            per_iter.append({"drafted": g, "accepted": a})
            # EOS / length truncation inside the block
            if not force_len:
                for i, t in enumerate(new):
                    if t in eos_ids:
                        new = new[:i + 1]
                        break
            new = new[:max_new_tokens - len(out)]
            seq += new
            out += new
            # --- rollback: keep pending + d1..da in target cache, i.e. new length = n0 - 1 + 1 + a = len(seq_old)+a
            tc.crop(n0 + a)
            dc.crop(min(n0 + a, dc.length))   # if a == g the draft never saw d_g, so it simply lags by one
            assert tc.length == n0 + a and (len(new) == a + 1 or done())
        t_dec = time.perf_counter() - t0
    stats = {"gamma": gamma, "iterations": len(per_iter), "drafted": drafted, "accepted": accepted,
             "acceptance_rate": accepted / drafted if drafted else 0.0,
             "mean_accepted_per_iter": accepted / len(per_iter) if per_iter else 0.0, "per_iter": per_iter,
             "kv_target_mb": kv_bytes(target, tc.length) / 2**20, "kv_draft_mb": kv_bytes(draft, dc.length) / 2**20}
    kv = stats["kv_target_mb"] + stats["kv_draft_mb"]
    return _finish(out, tokenizer, t_pre, t_dec, P, rss, kv, stats)


# ----------------------------------------------------------------------------------------------------------
# Library reference only.
# ----------------------------------------------------------------------------------------------------------
@torch.inference_mode()
def hf_generate(model, input_ids, max_new_tokens, eos_ids, tokenizer=None, force_len=False, assistant_model=None):
    from transformers import StoppingCriteria, StoppingCriteriaList

    P = input_ids.shape[1]
    first = {}

    class _T(StoppingCriteria):
        def __call__(self, ids, scores, **kw):
            first.setdefault("t", time.perf_counter())
            return torch.zeros(ids.shape[0], dtype=torch.bool)

    kw = dict(max_new_tokens=max_new_tokens, do_sample=False, eos_token_id=sorted(eos_ids),
              pad_token_id=sorted(eos_ids)[0], stopping_criteria=StoppingCriteriaList([_T()]),
              attention_mask=torch.ones_like(input_ids))
    if force_len:
        kw["min_new_tokens"] = max_new_tokens
    if assistant_model is not None:
        kw["assistant_model"] = assistant_model
    with _RSSPeak() as rss:
        t0 = time.perf_counter()
        g = model.generate(input_ids, **kw)
        t_all = time.perf_counter() - t0
    toks = g[0, P:].tolist()
    t_pre = min(first.get("t", t0 + t_all) - t0, t_all)
    return _finish(toks, tokenizer, t_pre, t_all - t_pre, P, rss, None, {"library": "transformers.generate"})
