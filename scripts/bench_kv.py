"""TGA-27: KV-cache benchmark (CPU). Modes: nocache (reference), kv (own cache), hf (library generate, reference).

Method: one process, merged fine-tuned target fp32. One tiny warm-up decode, then for every (mode, P, N) cell
`reps` timed repeats (force_len so every run generates exactly N tokens). Report median and IQR.
prefill_ms = time to first token; decode_ms = remaining N-1 tokens; decode tok/s = (N-1)/decode_s;
total tok/s = N/(prefill+decode). Peak RSS = in-call 5 ms sampler (process-wide, includes weights) and
delta vs RSS at call start. Analytical KV MB = 2*layers*kv_heads*head_dim*(P+N)*4 B.
Use --plot to rebuild CSV-merged figures from all results/kvcache/*/bench.csv.
"""
import argparse
import csv
import glob
import json
import statistics as st
import time

import torch

from kormat.common.paths import FIGURES, RESULTS
from kormat.common.runlog import finish_run, set_seed, start_run

ap = argparse.ArgumentParser()
ap.add_argument("--modes", default="kv")
ap.add_argument("--grid", default="128:16,128:64,128:128,512:16,512:64,512:128,1024:16,1024:64,1024:128")
ap.add_argument("--reps", type=int, default=3)
ap.add_argument("--name", default=None)
ap.add_argument("--plot", action="store_true")
a = ap.parse_args()


def q(xs):
    xs = sorted(xs)
    if len(xs) < 2:
        return 0.0
    qs = st.quantiles(xs, n=4, method="inclusive")
    return qs[2] - qs[0]


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = []
    for f in sorted(glob.glob(str(RESULTS / "kvcache" / "*" / "bench.csv"))):
        rows += list(csv.DictReader(open(f)))
    if not rows:
        return
    cells = {}
    for r in rows:
        cells.setdefault((r["mode"], int(r["P"]), int(r["N"])), []).append(r)
    med = lambda k, key: st.median(float(x[key]) for x in cells[k])
    FIGURES.mkdir(parents=True, exist_ok=True)
    col = {"nocache": "#c0392b", "kv": "#1f77b4", "hf": "#7f8c8d"}
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
    for mode in ("nocache", "kv", "hf"):
        for N, ls in ((16, ":"), (64, "--"), (128, "-")):
            ks = sorted(k for k in cells if k[0] == mode and k[2] == N)
            if not ks:
                continue
            xs = [k[1] for k in ks]
            lab = f"{mode} N={N}"
            ax[0].plot(xs, [med(k, "prefill_ms") / 1e3 for k in ks], ls, marker="o", color=col[mode], label=lab)
            ax[1].plot(xs, [med(k, "decode_tok_s") for k in ks], ls, marker="o", color=col[mode], label=lab)
            ax[2].plot(xs, [med(k, "total_tok_s") for k in ks], ls, marker="o", color=col[mode], label=lab)
    for x, t in zip(ax, ["prefill time (s)", "decode-only tok/s", "total tok/s"]):
        x.set_xlabel("prompt tokens P"); x.set_ylabel(t); x.set_xscale("log", base=2); x.grid(alpha=.3)
    ax[1].set_yscale("log"); ax[2].set_yscale("log")
    ax[2].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(FIGURES / "kv_speed.pdf"); fig.savefig(FIGURES / "kv_speed.png", dpi=150)
    # memory
    fig, ax = plt.subplots(figsize=(5, 3.4))
    for mode in ("nocache", "kv", "hf"):
        ks = sorted(k for k in cells if k[0] == mode)
        if ks:
            ax.scatter([k[1] + k[2] for k in ks], [med(k, "peak_rss_mb") / 1024 for k in ks], color=col[mode], label=mode)
    ks = sorted({k[1] + k[2] for k in cells})
    ax.set_xlabel("sequence length P+N"); ax.set_ylabel("peak process RSS (GB)"); ax.grid(alpha=.3); ax.legend()
    fig.tight_layout(); fig.savefig(FIGURES / "kv_memory.pdf"); fig.savefig(FIGURES / "kv_memory.png", dpi=150)
    print("figures written")


if a.plot:
    plot()
    raise SystemExit

from kormat.decoding import greedy_kv, greedy_nocache, hf_generate  # noqa: E402
from kormat.decoding.workloads import prompt_of_length  # noqa: E402
from kormat.modeling import load_target  # noqa: E402

set_seed(0)
run = start_run("kvcache", vars(a), name="bench" + (f"_{a.name}" if a.name else ""))
L = load_target(finetuned=True, merge=True, dtype=torch.float32)
tok, model = L.tokenizer, L.model
fn = {"nocache": greedy_nocache, "kv": greedy_kv,
      "hf": lambda m, i, n, e, force_len=True: hf_generate(m, i, n, e, force_len=force_len)}
w_ids, _ = prompt_of_length(tok, 64)
for m in a.modes.split(","):
    fn[m](model, w_ids[:, :48], 3, set(), force_len=True)   # warm-up (allocator, kernels)
rows, summ = [], []
prompts = {}
for cell in a.grid.split(","):
    P, N = (int(x) for x in cell.split(":"))
    if P not in prompts:
        prompts[P] = prompt_of_length(tok, P)
    ids, meta = prompts[P]
    for mode in a.modes.split(","):
        cr = []
        for rep in range(a.reps):
            r = fn[mode](model, ids, N, set(), force_len=True)
            dec_s = r.decode_ms / 1e3
            row = dict(mode=mode, P=meta["n_tokens"], N=N, rep=rep, prefill_ms=r.prefill_ms, decode_ms=r.decode_ms,
                       decode_tok_s=(N - 1) / dec_s if dec_s > 0 else 0, total_tok_s=r.tokens_per_s,
                       peak_rss_mb=r.peak_rss_mb, rss_delta_mb=r.peak_rss_mb - r.rss_start_mb,
                       kv_analytical_mb=__import__("kormat.decoding", fromlist=["kv_bytes"]).kv_bytes(model, meta["n_tokens"] + N) / 2**20)
            rows.append(row); cr.append(row)
        s = {"mode": mode, "P": meta["n_tokens"], "N": N}
        for key in ("prefill_ms", "decode_ms", "decode_tok_s", "total_tok_s", "peak_rss_mb", "rss_delta_mb"):
            v = [x[key] for x in cr]
            s[key + "_median"] = st.median(v); s[key + "_iqr"] = q(v)
        s["kv_analytical_mb"] = cr[0]["kv_analytical_mb"]
        summ.append(s)
        print(s, flush=True)
    with open(run / "bench.csv", "w", newline="") as f:
        w = csv.DictWriter(f, list(rows[0])); w.writeheader(); w.writerows(rows)
(run / "summary.json").write_text(json.dumps(summ, indent=1))
finish_run(run, {"summary": summ}, headline=f"kv bench modes={a.modes} cells={len(summ)}")
print("run dir", run)
