"""TGA-34: speculative greedy (own) vs target greedy (own KV) on real val prompts; HF assisted generation = reference.
Records exact-match, first divergence, and (on mismatch) the top1-top2 target margin at the divergence step."""
import argparse
import csv
import time

import torch

from kormat.common.runlog import finish_run, set_seed, start_run
from kormat.data import load_split
from kormat.decoding import greedy_kv, hf_generate, speculative_greedy
from kormat.decoding.workloads import record_prompt
from kormat.modeling import eos_ids, load_draft, load_target

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=8)
ap.add_argument("--gammas", default="1,2,4,8")
ap.add_argument("--max-new", type=int, default=64)
ap.add_argument("--hf", type=int, default=1, help="also run HF assisted generation reference")
ap.add_argument("--dtype", default="fp32", choices=["fp32", "bf16"])
ap.add_argument("--max-minutes", type=float, default=17)
a = ap.parse_args()
dt = {"fp32": torch.float32, "bf16": torch.bfloat16}[a.dtype]
set_seed(0)
run = start_run("speculative", vars(a), name=f"equiv_{a.dtype}")
T, D = load_target(True, True, dt), load_draft(dt)
tok = T.tokenizer
eos = eos_ids(tok, T.model)
val = load_split("val")
sel = val[:: max(1, len(val) // a.n)][: a.n]
gam = [int(g) for g in a.gammas.split(",")]
rows, t0 = [], time.time()
for pi, rec in enumerate(sel):
    ids = record_prompt(tok, rec)
    ref = greedy_kv(T.model, ids, a.max_new, eos, tok)
    variants = [(f"spec_g{g}", lambda g=g: speculative_greedy(T.model, D.model, ids, a.max_new, eos, g, tok)) for g in gam]
    if a.hf:
        variants.append(("hf_assisted", lambda: hf_generate(T.model, ids, a.max_new, eos, tok, assistant_model=D.model)))
    for name, f in variants:
        r = f()
        div = next((i for i in range(min(len(r.tokens), len(ref.tokens))) if r.tokens[i] != ref.tokens[i]), None)
        if div is None and len(r.tokens) != len(ref.tokens):
            div = min(len(r.tokens), len(ref.tokens))
        row = {"prompt": pi, "P": ids.shape[1], "method": name, "n_new_ref": len(ref.tokens), "n_new": len(r.tokens),
               "exact": div is None, "first_div": div, "alpha": r.stats.get("acceptance_rate", ""),
               "iters": r.stats.get("iterations", "")}
        if div is not None and div < len(ref.tokens):   # margin of the target's own greedy logits at divergence
            tr = []
            greedy_kv(T.model, ids, div + 1, set(), force_len=True, trace=tr)
            t2 = torch.topk(tr[div], 2).values
            row["margin_at_div"] = float(t2[0] - t2[1])
        rows.append(row)
        print(row, flush=True)
    if (time.time() - t0) / 60 > a.max_minutes:
        print("time budget reached", flush=True)
        break
keys = sorted({k for r in rows for k in r})
with open(run / "equivalence.csv", "w", newline="") as f:
    w = csv.DictWriter(f, keys); w.writeheader(); w.writerows(rows)
own = [r for r in rows if r["method"].startswith("spec")]
hf = [r for r in rows if r["method"] == "hf_assisted"]
m = {"own_exact": sum(r["exact"] for r in own), "own_total": len(own),
     "hf_exact": sum(r["exact"] for r in hf), "hf_total": len(hf), "rows": rows}
finish_run(run, m, headline=f"{a.dtype} own spec exact {m['own_exact']}/{m['own_total']}; hf assisted {m['hf_exact']}/{m['hf_total']}")
print("run dir", run)
