"""Route HF Qwen2 attention through tiled_attention by patching F.scaled_dot_product_attention (context manager)."""
from __future__ import annotations

import contextlib

import torch
import torch.nn.functional as F

from .tiled import tiled_attention


@contextlib.contextmanager
def tiled_sdpa_patch(block_q: int = 64, block_k: int = 64, variant: str = "kv_blocked", counter: dict | None = None):
    orig = F.scaled_dot_product_attention

    def patched(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False):
        assert dropout_p == 0.0
        mask = None
        if attn_mask is not None:
            # HF passes an additive float mask (0 keep / large-negative drop) or a bool mask.
            mask = attn_mask if attn_mask.dtype == torch.bool else (attn_mask == 0)
        # With attn_mask given HF sets is_causal=False; with is_causal=True and Lq==Lk, bottom-right == top-left.
        if is_causal:
            assert q.shape[2] == k.shape[2], "top-left is_causal differs from bottom-right when Lq != Lk"
        if counter is not None:
            counter["calls"] = counter.get("calls", 0) + 1
        return tiled_attention(q, k, v, causal=is_causal, attn_mask=mask, scale=scale, block_q=block_q,
                               block_k=block_k, variant=variant)

    F.scaled_dot_product_attention = patched
    try:
        yield
    finally:
        F.scaled_dot_product_attention = orig
