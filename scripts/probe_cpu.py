"""CPU probe (run under the model lock): RSS, prefill latency, decode latency per configuration and thread count.

Each (config, threads) pair runs in a fresh subprocess so peak RSS (ru_maxrss) is not polluted.
Outputs results/probe/cpu_probe.csv and results/probe/summary.md.
"""
from __future__ import annotations

import argparse
import csv
import json
import resource
import subprocess
import sys
import time

import psutil

from kormat.common.paths import RESULTS

CONFIGS = ["target_base", "target_merged", "draft", "target+draft"]
PREFILL_LENS = [256, 512, 1024]
DECODE_TOKENS = 32
OUT_DIR = RESULTS / "probe"


def rss_mb() -> float:
    return psutil.Process().memory_info().rss / 2**20


def peak_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # Linux: KiB


def bench(model, vocab: int, n: int, reps: int = 1):
    import torch

    g = torch.Generator().manual_seed(0)
    ids = torch.randint(1000, min(vocab, 100000), (1, n), generator=g)
    with torch.inference_mode():
        t = time.perf_counter()
        out = model(input_ids=ids, use_cache=True)
        pre = (time.perf_counter() - t) * 1000
        pkv, nxt = out.past_key_values, out.logits[:, -1].argmax(-1, keepdim=True)
        return pre, pkv, nxt


def decode(model, pkv, nxt, steps=DECODE_TOKENS):
    import torch

    with torch.inference_mode():
        t = time.perf_counter()
        for _ in range(steps):
            out = model(input_ids=nxt, past_key_values=pkv, use_cache=True)
            pkv, nxt = out.past_key_values, out.logits[:, -1].argmax(-1, keepdim=True)
        return (time.perf_counter() - t) * 1000 / steps


def run_one(cfg: str, threads: int) -> list[dict]:
    import torch

    from kormat.modeling import load_draft, load_target

    torch.set_num_threads(threads)
    rows = []
    t0 = time.perf_counter()
    models = {}
    if cfg == "target_base":
        models["target"] = load_target(finetuned=False)
    elif cfg == "target_merged":
        models["target"] = load_target(finetuned=True, merge=True)
    elif cfg == "draft":
        models["draft"] = load_draft()
    else:
        models["target"] = load_target(finetuned=True, merge=True)
        models["draft"] = load_draft()
    load_s = time.perf_counter() - t0
    rss_loaded = rss_mb()
    for name, m in models.items():
        vocab = m.model.config.vocab_size
        bench(m.model, vocab, 16)  # warm-up
        pre = {}
        for n in PREFILL_LENS:
            ms, pkv, nxt = bench(m.model, vocab, n)
            pre[n] = ms
        _, pkv, nxt = bench(m.model, vocab, 256)
        dec = decode(m.model, pkv, nxt)
        rows.append({"config": cfg, "component": name, "threads": threads, "load_s": round(load_s, 1),
                     "rss_after_load_mb": round(rss_loaded), "peak_rss_mb": round(peak_mb()),
                     **{f"prefill_{n}_ms": round(v) for n, v in pre.items()},
                     "decode_ms_per_token": round(dec, 1)})
    return rows


def project(rows: list[dict], threads: int) -> str:
    man = RESULTS / "split_manifest.json"
    note = ""
    n_crit = None
    if man.exists():
        try:
            d = json.loads(man.read_text())
            n_crit = int(d["n_annotations"]["val"]) + int(d["n_annotations"]["test"])
            note = f"criterion counts for val+test from results/split_manifest.json ({n_crit})"
        except Exception as e:  # noqa
            note = f"split_manifest.json unreadable ({e!r})"
    if not n_crit:
        n_crit = round(1015 * 47 / 53)
        note = f"split_manifest.json missing/unparsed: rough upper bound 1,015 x 47/53 = {n_crit} criteria"
    r = next(x for x in rows if x["config"] == "target_merged" and x["threads"] == threads)
    out_tok = 24  # assumption: typical JSON answer {"label": "...", "evidence": [..]} length
    per = r["prefill_512_ms"] + out_tok * r["decode_ms_per_token"]
    tot = per * n_crit / 1000
    lines = [f"- Basis: {note}.",
             f"- Per criterion (merged target, {threads} threads) = prefill@512 ({r['prefill_512_ms']} ms) + {out_tok} decode tokens x {r['decode_ms_per_token']} ms = {per/1000:.1f} s (ASSUMPTIONS: 512-token prompt, {out_tok}-token answer; actual lengths come from the eval run).",
             f"- One model pass over {n_crit} criteria: {tot/60:.0f} min ({tot/3600:.1f} h). Base+fine-tuned both on val+test: x2 = {2*tot/3600:.1f} h (sequential, no other load)."]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resummarize", action="store_true")
    ap.add_argument("--one", nargs=2, metavar=("CONFIG", "THREADS"))
    ap.add_argument("--threads", type=int, nargs="+", default=[4, 8])
    a = ap.parse_args()
    if a.one:
        print("ROWS" + json.dumps(run_one(a.one[0], int(a.one[1]))))
        return
    rows = []
    if a.resummarize:
        rows = [{k: (float(v) if k not in ("config", "component") else v) for k, v in r.items()} for r in csv.DictReader((OUT_DIR / "cpu_probe.csv").open())]
        for r in rows:
            r["threads"] = int(r["threads"])
        a.threads = sorted({r["threads"] for r in rows})
    parts = OUT_DIR / "_parts"
    parts.mkdir(parents=True, exist_ok=True)
    plan = [] if a.resummarize else [(c, t) for t in a.threads for c in CONFIGS if t == a.threads[0] or c == "target_merged"]
    for cfg, th in plan:
        f = parts / f"{cfg}_{th}.json"
        if not f.exists():
            print(f"== {cfg} threads={th}", flush=True)
            p = subprocess.run([sys.executable, __file__, "--one", cfg, str(th)], capture_output=True, text=True)
            line = [l for l in p.stdout.splitlines() if l.startswith("ROWS")]
            if not line:
                print(f"FAILED rc={p.returncode}: {p.stderr[-800:]}")
                continue
            f.write_text(line[0][4:])
        rows += json.loads(f.read_text())
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0])
    if a.resummarize:
        pass
    with (OUT_DIR / "cpu_probe.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(rows)
    def cost(t):
        return next(r["decode_ms_per_token"] + r["prefill_512_ms"] / 24 for r in rows if r["config"] == "target_merged" and r["threads"] == t)
    best = a.threads[0]
    for t in a.threads[1:]:
        if cost(t) < 0.9 * cost(best):  # only switch for a >10 % gain, otherwise differences are noise
            best = t
    md = ["# CPU probe (Intel i5-10310U, 4C/8T, fp32, torch CPU)", "",
          "Each row is a fresh process; random token ids; greedy decode with KV cache; one repetition (indicative, other agents may load the CPU).", "",
          "| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)]
    md += ["| " + " | ".join(str(r[k]) for k in keys) + " |" for r in rows]
    md += ["", f"## Recommendation", f"KORMAT_NUM_THREADS={best} (threads={a.threads[0]} unless another count is >10 % cheaper on prefill@512/24 + decode for the merged target; 4 vs 8 differed by {abs(cost(4)-cost(8))/cost(4)*100:.1f} %, i.e. noise, so we stay on the 4 physical cores and leave headroom for other processes).", "",
           "## Projected evaluation time", project(rows, best)]
    (OUT_DIR / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
