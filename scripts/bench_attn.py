"""CPU fp32 attention benchmark: runtime + peak extra memory (one subprocess per config) + block-size sweep + figures.

Memory method: a fresh subprocess builds q,k,v, records RSS, then a 0.5 ms sampling thread tracks peak RSS during
one call; peak_extra = peak_RSS - RSS_before_call (includes the output tensor and any L x L temporaries).
Compare with analytical score-matrix bytes B*H*L*L*4.
"""
import argparse, json, subprocess, sys, threading, time
from pathlib import Path

import torch

from kormat.common.paths import FIGURES, RESULTS
from kormat.common.runlog import capture_env, set_seed

OUT = RESULTS / "attention"
LS = [256, 512, 1024, 2048, 4096]
IMPLS = ["ref", "tiled_loops", "tiled_kv_blocked", "sdpa", "sdpa_math"]
D, H, B = 128, 12, 1


def worker(impl, L, block, causal):
    import psutil
    from kormat.attention.benchmark import get_impl, time_fn
    set_seed(0)
    q, k, v = (torch.randn(B, H, L, D) for _ in range(3))
    fn = get_impl(impl, causal, block, block)
    proc = psutil.Process()
    with torch.inference_mode():
        fn(q[:, :, :64], k[:, :, :64], v[:, :, :64])  # touch allocator / thread pool before baseline
        base = proc.memory_info().rss
        peak = [base]; stop = threading.Event()

        def poll():
            while not stop.is_set():
                peak[0] = max(peak[0], proc.memory_info().rss); time.sleep(0.0005)
        th = threading.Thread(target=poll); th.start()
        fn(q, k, v)
        stop.set(); th.join()
    mem = peak[0] - base
    t = time_fn(lambda: fn(q, k, v), 2, 5)
    print("RESULT" + json.dumps({"impl": impl, "L": L, "block": block, "causal": causal, "median_s": t["median_s"],
                                 "min_s": t["min_s"], "max_s": t["max_s"], "peak_extra_bytes": mem,
                                 "score_matrix_bytes": B * H * L * L * 4}))


def run(impl, L, block=64, causal=False):
    p = subprocess.run([sys.executable, __file__, "--worker", impl, str(L), str(block), str(int(causal))],
                       capture_output=True, text=True, timeout=900)
    for line in p.stdout.splitlines():
        if line.startswith("RESULT"):
            return json.loads(line[6:])
    return {"impl": impl, "L": L, "block": block, "error": p.stderr[-500:]}


def backends():
    from torch.nn.attention import SDPBackend, sdpa_kernel
    import torch.nn.functional as F
    q = torch.randn(1, 4, 128, 64)
    res = {}
    for name in ["MATH", "FLASH_ATTENTION", "EFFICIENT_ATTENTION", "CUDNN_ATTENTION"]:
        be = getattr(SDPBackend, name, None)
        if be is None:
            res[name] = "enum member absent"; continue
        try:
            with sdpa_kernel(be):
                F.scaled_dot_product_attention(q, q, q)
            res[name] = "works"
        except Exception as e:  # noqa
            res[name] = "unavailable: " + str(e).splitlines()[0][:100]
    return res


def main():
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    OUT.mkdir(parents=True, exist_ok=True); FIGURES.mkdir(parents=True, exist_ok=True)
    rows = []
    for L in LS:
        for impl in IMPLS:
            r = run(impl, L); rows.append(r); print(r, flush=True)
    sweep = []
    for blk in [16, 32, 64, 128, 256, 512]:
        for impl in ["tiled_loops", "tiled_kv_blocked"]:
            r = run(impl, 2048, blk); sweep.append(r); print(r, flush=True)
    res = {"config": {"d": D, "H": H, "B": B, "dtype": "float32", "causal": False, "warmup": 2, "repeats": 5,
                      "stat": "median", "block_default": 64, "threads": torch.get_num_threads()},
           "sdpa_backends_cpu": backends(), "runtime_memory": rows, "block_sweep_L2048": sweep, "env": capture_env()}
    (OUT / "bench.json").write_text(json.dumps(res, indent=2, default=str))
    import csv
    with (OUT / "bench.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r})); w.writeheader(); w.writerows(rows)
    col = {"ref": "C3", "tiled_loops": "C0", "tiled_kv_blocked": "C1", "sdpa": "C2", "sdpa_math": "C4"}
    for key, ylabel, fname in [("median_s", "median runtime (s)", "attn_runtime_vs_L.pdf"),
                               ("peak_extra_bytes", "peak extra memory (MiB)", "attn_memory_vs_L.pdf")]:
        fig, ax = plt.subplots(figsize=(5, 3.4))
        for impl in IMPLS:
            rr = [r for r in rows if r["impl"] == impl and key in r]
            ys = [r[key] / (2**20 if key != "median_s" else 1) for r in rr]
            ax.plot([r["L"] for r in rr], ys, "o-", color=col[impl], label=impl)
        if key != "median_s":
            ax.plot(LS, [B * H * L * L * 4 / 2**20 for L in LS], "k--", label="L x L fp32 scores (analytical)")
        ax.set_xscale("log", base=2); ax.set_yscale("log"); ax.set_xlabel("sequence length L"); ax.set_ylabel(ylabel)
        ax.set_title(f"CPU fp32, d={D}, H={H}, B={B}"); ax.legend(fontsize=7); ax.grid(alpha=.3)
        fig.tight_layout(); fig.savefig(FIGURES / fname); plt.close(fig)
    fig, ax = plt.subplots(figsize=(5, 3.4))
    for impl in ["tiled_loops", "tiled_kv_blocked"]:
        rr = [r for r in sweep if r["impl"] == impl and "median_s" in r]
        ax.plot([r["block"] for r in rr], [r["median_s"] for r in rr], "o-", color=col[impl], label=impl)
    ref = next((r for r in rows if r["impl"] == "sdpa" and r["L"] == 2048), None)
    if ref: ax.axhline(ref["median_s"], color="C2", ls="--", label="torch SDPA")
    ax.set_xscale("log", base=2); ax.set_xlabel("block size (block_q = block_k)"); ax.set_ylabel("median runtime (s)")
    ax.set_title("Block-size sweep, L=2048"); ax.legend(fontsize=7); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(FIGURES / "attn_runtime_vs_block.pdf")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        worker(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), bool(int(sys.argv[5])))
    else:
        main()
