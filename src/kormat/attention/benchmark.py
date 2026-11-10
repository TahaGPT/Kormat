"""CPU timing helpers. run_micro() is the single-shape quick readout used by the app's lab panel."""
from __future__ import annotations

import statistics
import time

import torch
import torch.nn.functional as F

from .reference import attention_ref
from .tiled import tiled_attention

IMPLS = ("ref", "tiled_loops", "tiled_kv_blocked", "sdpa", "sdpa_math")


def get_impl(name: str, causal: bool = False, block_q: int = 64, block_k: int = 64):
    if name == "ref":
        return lambda q, k, v: attention_ref(q, k, v, causal=causal)
    if name == "tiled_loops":
        return lambda q, k, v: tiled_attention(q, k, v, causal=causal, block_q=block_q, block_k=block_k, variant="loops")
    if name == "tiled_kv_blocked":
        return lambda q, k, v: tiled_attention(q, k, v, causal=causal, block_q=block_q, block_k=block_k, variant="kv_blocked")
    if name == "sdpa":
        return lambda q, k, v: F.scaled_dot_product_attention(q, k, v, is_causal=causal)
    if name == "sdpa_math":
        from torch.nn.attention import SDPBackend, sdpa_kernel

        def f(q, k, v):
            with sdpa_kernel(SDPBackend.MATH):
                return F.scaled_dot_product_attention(q, k, v, is_causal=causal)
        return f
    raise ValueError(name)


def time_fn(fn, warmup: int = 2, repeats: int = 5):
    with torch.inference_mode():
        for _ in range(warmup):
            fn()
        ts = []
        for _ in range(repeats):
            t = time.perf_counter(); fn(); ts.append(time.perf_counter() - t)
    return {"median_s": statistics.median(ts), "min_s": min(ts), "max_s": max(ts), "times_s": ts}


def run_micro(L: int = 512, d: int = 128, H: int = 12, impls=IMPLS, causal: bool = False, B: int = 1,
              block: int = 64, warmup: int = 1, repeats: int = 3, seed: int = 0) -> dict:
    """Time each impl at one shape and report max abs error vs a float64 attention_ref. Returns {impl: {...}}."""
    g = torch.Generator().manual_seed(seed)
    q, k, v = (torch.randn(B, H, L, d, generator=g) for _ in range(3))
    truth = attention_ref(q.double(), k.double(), v.double(), causal=causal, accum_dtype=torch.float64)
    out = {}
    for name in impls:
        fn = get_impl(name, causal, block, block)
        t = time_fn(lambda: fn(q, k, v), warmup, repeats)
        with torch.inference_mode():
            err = (fn(q, k, v).double() - truth).abs().max().item()
        out[name] = {"median_ms": t["median_s"] * 1e3, "max_abs_err": err}
    out["_meta"] = {"L": L, "d": d, "H": H, "B": B, "causal": causal, "block": block, "dtype": "float32",
                    "score_matrix_MB": B * H * L * L * 4 / 2**20}
    return out
