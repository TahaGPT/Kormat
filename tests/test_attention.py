import pytest
import torch
import torch.nn.functional as F

from kormat.attention import attention_ref, tiled_attention
from kormat.attention.benchmark import run_micro
from kormat.attention.validate import MASKS, run_case


def _rand(B=1, H=4, Hkv=4, Lq=33, Lk=33, d=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(B, H, Lq, d, generator=g), torch.randn(B, Hkv, Lk, d, generator=g),
            torch.randn(B, Hkv, Lk, d, generator=g))


def test_ref_matches_torch_sdpa_with_gqa_and_bottom_right_causal():
    q, k, v = _rand(2, 6, 2, 5, 40)
    causal = torch.arange(40)[None, :] <= torch.arange(5)[:, None] + 35
    exp = F.scaled_dot_product_attention(q, k.repeat_interleave(3, 1), v.repeat_interleave(3, 1), attn_mask=causal)
    assert torch.allclose(attention_ref(q, k, v, causal=True), exp, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("variant", ["loops", "kv_blocked"])
@pytest.mark.parametrize("L,bs", [(1, 16), (7, 16), (65, 32), (127, 64), (64, 128)])
@pytest.mark.parametrize("mask", MASKS)
def test_tiled_fp32_vs_f64(variant, L, bs, mask):
    e = run_case("tiled_" + variant, 2, 12, 2, L, L, 32, torch.float32, mask, bs, bs)
    assert e["pass"], e


@pytest.mark.parametrize("variant", ["loops", "kv_blocked"])
@pytest.mark.parametrize("dt", [torch.bfloat16, torch.float16])
def test_tiled_half(variant, dt):
    e = run_case("tiled_" + variant, 1, 4, 4, 65, 65, 32, dt, "causal+padding", 16, 32)
    assert e["pass"], e


@pytest.mark.parametrize("variant", ["loops", "kv_blocked"])
@pytest.mark.parametrize("lq,lk", [(1, 777), (5, 300)])
def test_lq_ne_lk(variant, lq, lk):
    for mask in MASKS:
        assert run_case("tiled_" + variant, 1, 4, 2, lq, lk, 16, torch.float32, mask, 16, 64)["pass"]


def test_fully_masked_row_is_zero_not_nan():
    q, k, v = _rand()
    m = torch.ones(1, 1, 33, 33, dtype=torch.bool); m[..., 3, :] = False
    for fn in (lambda: attention_ref(q, k, v, attn_mask=m),
               lambda: tiled_attention(q, k, v, attn_mask=m, block_q=8, block_k=8)):
        o = fn()
        assert not torch.isnan(o).any() and (o[:, :, 3] == 0).all()


def test_lse_and_stability_at_large_logits():
    q, k, v = _rand(1, 2, 2, 40, 40, 16)
    sc = 125.0  # logits x 1000 vs 1/sqrt(16)=0.25 * 1000 ... naive exp overflows
    s = (q @ k.transpose(-1, -2)) * sc
    naive = torch.exp(s) / torch.exp(s).sum(-1, keepdim=True)
    assert torch.isnan(naive).any()
    o, lse = tiled_attention(q, k, v, scale=sc, block_q=8, block_k=8, return_lse=True)
    ref = attention_ref(q.double(), k.double(), v.double(), scale=sc, accum_dtype=torch.float64)
    assert torch.allclose(o.double(), ref, atol=1e-4) and torch.isfinite(lse).all()
    assert torch.allclose(lse.double(), torch.logsumexp(s.double(), -1), rtol=1e-5, atol=1e-3)


def test_run_micro_fast():
    r = run_micro(L=128, H=2, d=32, impls=["ref", "tiled_loops", "tiled_kv_blocked", "sdpa"])
    assert all(r[n]["max_abs_err"] < 1e-5 for n in ("ref", "tiled_loops", "tiled_kv_blocked", "sdpa"))


def test_model_integration_tiny():
    from kormat.attention.integration import tiled_sdpa_patch
    from kormat.modeling import tiny_qwen
    m = tiny_qwen(); ids = torch.randint(1, 1000, (1, 21))
    with torch.inference_mode():
        base = m(ids).logits; g0 = m.generate(ids, max_new_tokens=6, do_sample=False, pad_token_id=0)
        cnt = {}
        with tiled_sdpa_patch(8, 8, "loops", cnt):
            t = m(ids).logits; g1 = m.generate(ids, max_new_tokens=6, do_sample=False, pad_token_id=0)
    assert cnt["calls"] > 0 and torch.equal(g0, g1) and (base - t).abs().max() < 1e-4
