"""Benchmark/equivalence workloads built from real train/val notes (never the sealed test split)."""
from __future__ import annotations

import torch

from kormat.data import load_split
from kormat.prompting.template import build_messages, render_prompt


def prompt_of_length(tokenizer, target_len: int, split: str = "val", record_idx: int = 0):
    """Chat-templated prompt (frozen contract) with >= target_len tokens, padded by WHOLE sentences.

    Sentences are taken from consecutive val records (record_idx, record_idx+1, ...) in order and numbered
    0..k-1; the criterion is that of the first record. Stops at the first sentence count whose rendered prompt
    reaches target_len (so length is in [target_len, target_len + one sentence)).
    Returns (input_ids (1,P), meta).
    """
    recs = load_split(split)
    sents: list[str] = []
    for r in recs[record_idx:] + recs[:record_idx]:
        for _, s in sorted(r["sentences"].items()):
            sents.append(s)
    r0 = recs[record_idx]
    lo, hi = 1, len(sents)

    def enc(k):
        msgs = build_messages(dict(enumerate(sents[:k])), r0["criterion_type"], r0["criterion_text"])
        return tokenizer(render_prompt(tokenizer, msgs), return_tensors="pt", add_special_tokens=False).input_ids

    while lo < hi:                      # smallest k with len >= target (length is monotone in k)
        mid = (lo + hi) // 2
        if enc(mid).shape[1] >= target_len:
            hi = mid
        else:
            lo = mid + 1
    ids = enc(lo)
    return ids, {"target_len": target_len, "n_tokens": ids.shape[1], "n_sentences": lo, "record": record_idx}


def record_prompt(tokenizer, rec: dict) -> torch.Tensor:
    msgs = build_messages(rec["sentences"], rec["criterion_type"], rec["criterion_text"])
    return tokenizer(render_prompt(tokenizer, msgs), return_tensors="pt", add_special_tokens=False).input_ids
