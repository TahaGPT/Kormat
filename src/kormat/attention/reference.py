"""Reference scaled dot-product attention (materialises the full Lq x Lk score matrix).

Shapes: q (B,H,Lq,d); k,v (B,Hkv,Lk,d) with H % Hkv == 0 (grouped-query attention).
Causal mask is aligned bottom-right: query i may see key j iff j <= i + (Lk - Lq),
which is the correct rule when a short query block attends to a longer cached prefix.
attn_mask is bool (True = keep), broadcastable to (B,H,Lq,Lk).
"""
from __future__ import annotations

import math

import torch


def expand_kv(x: torch.Tensor, n_heads: int) -> torch.Tensor:
    """Repeat KV heads so (B,Hkv,L,d) -> (B,H,L,d) (head h uses kv head h // group)."""
    hkv = x.shape[1]
    if hkv == n_heads:
        return x
    if n_heads % hkv:
        raise ValueError(f"H={n_heads} not divisible by Hkv={hkv}")
    return x.repeat_interleave(n_heads // hkv, dim=1)


def as_4d_mask(attn_mask: torch.Tensor) -> torch.Tensor:
    if attn_mask.dtype != torch.bool:
        raise TypeError("attn_mask must be bool (True = keep)")
    while attn_mask.dim() < 4:
        attn_mask = attn_mask.unsqueeze(0)
    return attn_mask


def causal_keep(lq: int, lk: int, q0: int = 0, q1: int | None = None, k0: int = 0, k1: int | None = None,
                device=None) -> torch.Tensor:
    """Bool (q1-q0, k1-k0) block of the bottom-right-aligned causal mask."""
    q1 = lq if q1 is None else q1
    k1 = lk if k1 is None else k1
    qi = torch.arange(q0, q1, device=device).unsqueeze(1)
    kj = torch.arange(k0, k1, device=device).unsqueeze(0)
    return kj <= qi + (lk - lq)


def attention_ref(q, k, v, causal: bool = False, attn_mask=None, scale: float | None = None,
                  accum_dtype: torch.dtype = torch.float32):
    B, H, Lq, d = q.shape
    Lk = k.shape[2]
    scale = 1.0 / math.sqrt(d) if scale is None else scale
    qf = q.to(accum_dtype)
    kf = expand_kv(k, H).to(accum_dtype)
    vf = expand_kv(v, H).to(accum_dtype)
    s = torch.matmul(qf, kf.transpose(-1, -2))  # (B,H,Lq,Lk): the O(L^2) buffer
    s.mul_(scale)
    keep = None
    if causal:
        keep = causal_keep(Lq, Lk, device=q.device)
    if attn_mask is not None:
        am = as_4d_mask(attn_mask)
        keep = am if keep is None else (am & keep)
    if keep is not None:
        s.masked_fill_(~keep, float("-inf"))
    m = s.amax(dim=-1, keepdim=True)
    m = torch.where(torch.isfinite(m), m, torch.zeros_like(m))  # fully masked row: avoid -inf - -inf
    s.sub_(m)
    s.exp_()  # masked entries: exp(-inf) = 0
    l = s.sum(dim=-1, keepdim=True)
    p = s / torch.where(l > 0, l, torch.ones_like(l))  # fully masked row -> p = 0 -> output 0
    return torch.matmul(p, vf).to(q.dtype)
