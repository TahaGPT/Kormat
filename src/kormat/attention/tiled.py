"""FlashAttention-style tiled attention with online softmax (Dao et al., Algorithm 1), pure PyTorch, CPU.

For each query block we keep running statistics over the key blocks seen so far:
    m : running row max of scores,  l : running normaliser sum_j exp(s_j - m),  O : unnormalised output.
On a new key block with scores S, m' = max(m, rowmax(S)), P = exp(S - m'),
    l' = exp(m - m') * l + rowsum(P),   O' = exp(m - m') * O + P @ V,
and the final output is O / l. The L x L matrix is never materialised (only block_q x block_k tiles).

variant="loops":     explicit Python loops over query blocks (outer) and key blocks (inner), as in the paper.
variant="kv_blocked": all queries at once, loop only over key blocks (fewer, larger ops; O(Lq*block_k) tile).
All accumulation (m, l, O, matmuls) is fp32; output is cast back to q.dtype.
"""
from __future__ import annotations

import math

import torch

from .reference import as_4d_mask, expand_kv

NEG_INF = float("-inf")


def _slice_mask(am: torch.Tensor, q0, q1, k0, k1) -> torch.Tensor:
    qs = slice(q0, q1) if am.shape[-2] > 1 else slice(None)
    ks = slice(k0, k1) if am.shape[-1] > 1 else slice(None)
    return am[..., qs, ks]


def _tile_update(qb, kb, vb, m, l, o, scale, keep):
    """One online-softmax step. qb (B,H,bq,d), kb/vb (B,H,bk,d); m,l (B,H,bq,1); o (B,H,bq,d). In fp32."""
    s = torch.matmul(qb, kb.transpose(-1, -2)) * scale
    if keep is not None:
        s = s.masked_fill(~keep, NEG_INF)
    m_new = torch.maximum(m, s.amax(dim=-1, keepdim=True))
    m_safe = torch.where(torch.isfinite(m_new), m_new, torch.zeros_like(m_new))  # guard -inf - -inf = NaN
    alpha = torch.exp(m - m_safe)  # exp(-inf) = 0 on the first block / still-empty rows
    p = torch.exp(s - m_safe)
    l = alpha * l + p.sum(dim=-1, keepdim=True)
    o = alpha * o + torch.matmul(p, vb)
    return m_new, l, o


def _finish(o, l, m):
    ok = l > 0
    out = o / torch.where(ok, l, torch.ones_like(l))  # fully masked rows -> 0
    lse = torch.where(ok, m + torch.log(torch.where(ok, l, torch.ones_like(l))), torch.full_like(l, NEG_INF))
    return out, lse.squeeze(-1)


def tiled_attention(q, k, v, causal: bool = False, attn_mask=None, scale: float | None = None,
                    block_q: int = 64, block_k: int = 64, variant: str = "loops", return_lse: bool = False):
    if variant not in ("loops", "kv_blocked"):
        raise ValueError(variant)
    if block_q < 1 or block_k < 1:
        raise ValueError("block sizes must be >= 1")
    B, H, Lq, d = q.shape
    Lk = k.shape[2]
    scale = 1.0 / math.sqrt(d) if scale is None else scale
    off = Lk - Lq  # bottom-right causal alignment
    am = as_4d_mask(attn_mask) if attn_mask is not None else None
    qf = q.float()
    dev = q.device

    def keep_block(q0, q1, k0, k1):
        keep = None
        if causal:
            qi = torch.arange(q0, q1, device=dev).unsqueeze(1)
            kj = torch.arange(k0, k1, device=dev).unsqueeze(0)
            if k1 - 1 > q0 + off:  # block not fully below the diagonal -> needs masking
                keep = kj <= qi + off
        if am is not None:
            mb = _slice_mask(am, q0, q1, k0, k1)
            keep = mb if keep is None else (mb & keep)
        return keep

    def kv_block(k0, k1):
        return (expand_kv(k[:, :, k0:k1], H).float(), expand_kv(v[:, :, k0:k1], H).float())

    out = torch.empty(B, H, Lq, d, dtype=torch.float32, device=dev)
    lse = torch.empty(B, H, Lq, dtype=torch.float32, device=dev)

    if variant == "loops":
        for q0 in range(0, Lq, block_q):
            q1 = min(q0 + block_q, Lq)
            qb = qf[:, :, q0:q1]
            m = torch.full((B, H, q1 - q0, 1), NEG_INF, device=dev)
            l = torch.zeros(B, H, q1 - q0, 1, device=dev)
            o = torch.zeros(B, H, q1 - q0, d, device=dev)
            k_end = min(Lk, q1 + off) if causal else Lk  # causal block skipping
            for k0 in range(0, max(k_end, 0), block_k):
                k1 = min(k0 + block_k, Lk)
                kb, vb = kv_block(k0, k1)
                m, l, o = _tile_update(qb, kb, vb, m, l, o, scale, keep_block(q0, q1, k0, k1))
            out[:, :, q0:q1], lse[:, :, q0:q1] = _finish(o, l, m)
    else:
        m = torch.full((B, H, Lq, 1), NEG_INF, device=dev)
        l = torch.zeros(B, H, Lq, 1, device=dev)
        o = torch.zeros(B, H, Lq, d, device=dev)
        k_end = min(Lk, Lq + off) if causal else Lk
        for k0 in range(0, max(k_end, 0), block_k):
            k1 = min(k0 + block_k, Lk)
            kb, vb = kv_block(k0, k1)
            m, l, o = _tile_update(qf, kb, vb, m, l, o, scale, keep_block(0, Lq, k0, k1))
        out, lse = _finish(o, l, m)

    out = out.to(q.dtype)
    return (out, lse) if return_lse else out
