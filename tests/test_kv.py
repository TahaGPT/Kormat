import pytest
import torch

from kormat.decoding import PreallocKVCache, greedy_kv, greedy_nocache, hf_generate
from kormat.decoding.api import _fwd
from kormat.modeling import tiny_qwen

V = 1000


def _ids(P, seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(5, V, (1, P), generator=g)


@pytest.mark.parametrize("P,N", [(8, 12), (40, 30), (1, 5)])
def test_kv_equals_nocache(P, N):
    m = tiny_qwen(0, V)
    x = _ids(P)
    a = greedy_nocache(m, x, N, set(), force_len=True)
    b = greedy_kv(m, x, N, set(), force_len=True)
    assert a.tokens == b.tokens and len(a.tokens) == N
    assert b.stats["cache_update_calls"] == m.config.num_hidden_layers * N   # prefill + (N-1) decodes
    assert b.kv_cache_mb is not None


def test_kv_equals_hf_generate():
    m = tiny_qwen(0, V)
    x = _ids(20)
    assert greedy_kv(m, x, 16, set(), force_len=True).tokens == hf_generate(m, x, 16, {V + 5}, force_len=True).tokens


def test_eos_stops_and_is_included():
    m = tiny_qwen(0, V)
    x = _ids(10)
    full = greedy_kv(m, x, 10, set(), force_len=True).tokens
    eos = full[3]
    first = full.index(eos)
    r = greedy_kv(m, x, 10, {eos})
    assert r.tokens == full[:first + 1]
    assert greedy_nocache(m, x, 10, {eos}).tokens == r.tokens


def test_cache_is_really_used_and_growth():
    m = tiny_qwen(0, V)
    x = _ids(7)
    c = PreallocKVCache.for_model(m, 1, 32)
    _fwd(m, x, c)
    assert c.update_calls == 2 and c.length == 7
    _fwd(m, torch.tensor([[3]]), c)
    assert c.length == 8 and c.update_calls == 4
    k, v = c.k[0], c.v[0]
    assert k.shape == (1, 2, 32, 16)   # preallocated: (B, kv_heads, max_len, head_dim)
    c.crop(5)
    assert c.length == 5
    assert c.nbytes_used() == 2 * 2 * 1 * 2 * 16 * 5 * 4


def test_cache_logits_match_full_forward():
    """Logits of the incremental path equal one full forward at every position (positions/mask correct)."""
    m = tiny_qwen(0, V)
    x = _ids(12)
    with torch.inference_mode():
        full = m(x).logits[0]
        c = PreallocKVCache.for_model(m, 1, 16)
        rows = [_fwd(m, x[:, :5], c, last_only=False)]
        for i in range(5, 12):
            rows.append(_fwd(m, x[:, i:i + 1], c, last_only=False))
    inc = torch.cat(rows)
    assert (inc - full).abs().max() < 1e-4


def test_wrong_position_breaks():
    """Negative control: wrong position_ids give different logits, so the equivalence test is not vacuous."""
    m = tiny_qwen(0, V)
    x = _ids(12)
    with torch.inference_mode():
        c = PreallocKVCache.for_model(m, 1, 16)
        _fwd(m, x[:, :11], c)
        good = _fwd(m, x[:, 11:], c)
        c2 = PreallocKVCache.for_model(m, 1, 16)
        _fwd(m, x[:, :11], c2)
        pos = torch.tensor([11])
        bad = m(input_ids=x[:, 11:], attention_mask=torch.ones(1, 12, dtype=torch.long), position_ids=torch.tensor([[3]]),
                past_key_values=c2, use_cache=True, cache_position=pos).logits[0]
    assert (good - bad).abs().max() > 1e-3
