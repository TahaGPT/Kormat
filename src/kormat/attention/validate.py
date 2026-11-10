"""Numerical validation helpers shared by scripts/attn_validate.py and the tests."""
from __future__ import annotations

import torch

from .reference import attention_ref
from .tiled import tiled_attention

TOL = {  # (atol, rtol); |a-b| <= atol + rtol*|b|
    torch.float32: (1e-5, 1e-4),
    torch.float16: (2e-3, 2e-3),   # fp16 output rounding is <= 2^-11 relative
    torch.bfloat16: (2e-3, 8e-3),  # bf16 output rounding is <= 2^-9 ~ 2e-3 relative, one bf16 ulp = 2^-7 ~ 8e-3
}
MASKS = ("none", "causal", "causal+padding", "fully_masked_row")


def make_inputs(B, H, Hkv, Lq, Lk, d, dtype, seed=0):
    g = torch.Generator().manual_seed(seed)
    mk = lambda *s: torch.randn(*s, generator=g).to(dtype)
    return mk(B, H, Lq, d), mk(B, Hkv, Lk, d), mk(B, Hkv, Lk, d)


def make_mask(kind, B, Lq, Lk):
    """Returns (causal, attn_mask)."""
    if kind == "none":
        return False, None
    if kind == "causal":
        return True, None
    if kind == "causal+padding":
        lens = [max(1, Lk - 3), max(1, Lk // 2)][:B] + [Lk] * max(0, B - 2)
        pad = torch.zeros(B, 1, 1, Lk, dtype=torch.bool)
        for b, n in enumerate(lens):
            pad[b, ..., :n] = True
        return True, pad
    if kind == "fully_masked_row":
        m = torch.ones(1, 1, Lq, Lk, dtype=torch.bool)
        m[..., Lq // 2, :] = False
        return False, m
    raise ValueError(kind)


def truth(q, k, v, causal, mask):
    return attention_ref(q.double(), k.double(), v.double(), causal=causal, attn_mask=mask,
                         accum_dtype=torch.float64)


def errors(out, ref, dtype):
    o, r = out.double(), ref.double()
    diff = (o - r).abs()
    atol, rtol = TOL[dtype]
    return {
        "max_abs": diff.max().item(), "mean_abs": diff.mean().item(),
        "max_rel": (diff / r.abs().clamp_min(1e-6)).max().item(),  # relative to |truth| floored at 1e-6
        "has_nan": bool(torch.isnan(out.float()).any().item()),
        "pass": bool(not torch.isnan(out.float()).any() and (diff <= atol + rtol * r.abs()).all().item()),
        "atol": atol, "rtol": rtol,
    }


def run_case(impl, B, H, Hkv, Lq, Lk, d, dtype, mask_kind, bq=64, bk=64, seed=0):
    q, k, v = make_inputs(B, H, Hkv, Lq, Lk, d, dtype, seed)
    causal, mask = make_mask(mask_kind, B, Lq, Lk)
    ref = truth(q, k, v, causal, mask)  # inputs already rounded to `dtype`, then upcast: isolates kernel error
    if impl == "ref":
        out = attention_ref(q, k, v, causal=causal, attn_mask=mask)
    else:
        out = tiled_attention(q, k, v, causal=causal, attn_mask=mask, block_q=bq, block_k=bk,
                              variant="loops" if impl == "tiled_loops" else "kv_blocked")
    e = errors(out, ref, dtype)
    if mask_kind == "fully_masked_row":
        e["masked_row_zero"] = bool((out[:, :, Lq // 2].float() == 0).all().item())
        e["pass"] = e["pass"] and e["masked_row_zero"]
    return e
