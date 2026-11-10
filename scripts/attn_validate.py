"""Numerical validation grid: tiled/ref attention vs float64 ground truth -> results/attention/validation.csv + summary.json."""
import csv, json, sys, time, itertools
from pathlib import Path

import torch

from kormat.attention.validate import MASKS, run_case
from kormat.attention.reference import attention_ref
from kormat.common.paths import RESULTS
from kormat.common.runlog import capture_env, set_seed

OUT = RESULTS / "attention"
OUT.mkdir(parents=True, exist_ok=True)
IMPLS = ["ref", "tiled_loops", "tiled_kv_blocked"]


def dtypes():
    ds = [torch.float32, torch.bfloat16]
    fp16_ok, note = True, ""
    try:  # does CPU fp16 work for the ops we use?
        x = torch.randn(1, 1, 4, 8).half()
        attention_ref(x, x, x)
    except Exception as e:  # noqa
        fp16_ok, note = False, repr(e)
    if fp16_ok:
        ds.append(torch.float16)
    return ds, fp16_ok, note


def grid(ds):
    cases = []  # (B,H,Hkv,Lq,Lk,d,dtype,mask,bq,bk)
    # A: sequence-length sweep, all masks/dtypes
    for L, m, dt in itertools.product([1, 7, 16, 63, 64, 65, 127, 256, 513, 1024], MASKS, ds):
        cases.append((1, 4, 4, L, L, 64, dt, m, 32, 32))
    # B: Lq != Lk
    for (lq, lk), m, dt, bs in itertools.product([(1, 777), (5, 300)], MASKS, ds, [16, 64]):
        cases.append((2, 12, 2, lq, lk, 64, dt, m, bs, bs))
    # C: d x B x heads x blocks x masks x dtypes at L=127
    for d, B, (H, Hkv), bs, m, dt in itertools.product([64, 128], [1, 2], [(4, 4), (12, 2)], [16, 32, 64, 128], MASKS, ds):
        cases.append((B, H, Hkv, 127, 127, d, dt, m, bs, bs))
    # D: larger L, asymmetric blocks, d=128, GQA
    for bq, bk in [(16, 128), (128, 16), (64, 64), (32, 128)]:
        for m in ("none", "causal+padding"):
            cases.append((2, 12, 2, 513, 513, 128, torch.float32, m, bq, bk))
    return cases


def main():
    set_seed(0)
    ds, fp16_ok, fp16_note = dtypes()
    cases = grid(ds)
    rows = []
    t0 = time.time()
    for i, (B, H, Hkv, Lq, Lk, d, dt, m, bq, bk) in enumerate(cases):
        for impl in IMPLS:
            e = run_case(impl, B, H, Hkv, Lq, Lk, d, dt, m, bq, bk)
            rows.append({"impl": impl, "dtype": str(dt).replace("torch.", ""), "B": B, "H": H, "Hkv": Hkv, "Lq": Lq,
                         "Lk": Lk, "d": d, "mask": m, "block_q": bq, "block_k": bk, **e})
        if i % 200 == 0:
            print(i, len(cases), f"{time.time()-t0:.0f}s", flush=True)
    with (OUT / "validation.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) + ['masked_row_zero'], restval='')
        w.writeheader(); w.writerows(rows)
    summ = {"n_rows": len(rows), "n_pass": sum(r["pass"] for r in rows), "n_fail": sum(not r["pass"] for r in rows),
            "fp16_supported_on_cpu": fp16_ok, "fp16_note": fp16_note,
            "tolerances": {"float32": [1e-5, 1e-4], "float16": [2e-3, 2e-3], "bfloat16": [2e-3, 8e-3]},
            "tolerance_rule": "|out - truth| <= atol + rtol*|truth|; truth = attention_ref in float64 on the same (already dtype-rounded) inputs",
            "by_dtype": {}, "failures": [r for r in rows if not r["pass"]][:20], "env": capture_env()}
    for dt in sorted({r["dtype"] for r in rows}):
        rr = [r for r in rows if r["dtype"] == dt]
        summ["by_dtype"][dt] = {"rows": len(rr), "pass": sum(r["pass"] for r in rr),
                                "max_abs": max(r["max_abs"] for r in rr),
                                "mean_abs_mean": sum(r["mean_abs"] for r in rr) / len(rr),
                                "max_rel": max(r["max_rel"] for r in rr)}
    # stability demo
    torch.manual_seed(0)
    q, k, v = (torch.randn(1, 4, 128, 64) for _ in range(3))
    sc = 1000.0 / 8.0  # logits x1000 relative to the usual 1/sqrt(d)
    s = (q @ k.transpose(-1, -2)) * sc
    naive = (torch.exp(s) / torch.exp(s).sum(-1, keepdim=True)) @ v
    ours = tiled_attention_demo(q, k, v, sc)
    ref = attention_ref(q.double(), k.double(), v.double(), scale=sc, accum_dtype=torch.float64)
    summ["stability_demo"] = {
        "logit_scale_factor": 1000.0, "max_abs_logit": s.abs().max().item(),
        "naive_softmax_has_nan": bool(torch.isnan(naive).any()), "naive_softmax_has_inf": bool(torch.isinf(naive).any()),
        "tiled_has_nan": bool(torch.isnan(ours).any()), "tiled_max_abs_err_vs_f64": (ours.double() - ref).abs().max().item()}
    (OUT / "summary.json").write_text(json.dumps(summ, indent=2, default=str))
    print(json.dumps({k: summ[k] for k in ("n_rows", "n_pass", "n_fail", "by_dtype", "stability_demo")}, indent=1))
    print("fp16:", fp16_ok)
    return 0 if summ["n_fail"] == 0 else 1


def tiled_attention_demo(q, k, v, sc):
    from kormat.attention.tiled import tiled_attention
    return tiled_attention(q, k, v, scale=sc, block_q=32, block_k=32)


if __name__ == "__main__":
    sys.exit(main())
