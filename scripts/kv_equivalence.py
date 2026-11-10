"""TGA-26: token/logit equivalence of no-cache vs own-KV greedy decoding on the merged fine-tuned target.

For each input length P we decode the LONGEST requested N once per path (greedy is prefix-closed, so the first
16/64 tokens of a 128-token run are exactly the 16/64-token runs) and score every N prefix.
Per cell: exact match, first divergence index, max |dlogit| over steps up to and including the divergence
(identical inputs there), and top1-top2 margin of the no-cache logits at the divergence step.
"""
import argparse
import csv
import json
import sys
import time

import torch

from kormat.common.runlog import finish_run, set_seed, start_run
from kormat.decoding import greedy_kv, greedy_nocache
from kormat.decoding.workloads import prompt_of_length
from kormat.modeling import eos_ids, load_target

ap = argparse.ArgumentParser()
ap.add_argument("--grid", default="128:128,512:64,1024:16", help="P:Nmax,...")
ap.add_argument("--ns", default="16,64,128")
ap.add_argument("--dtype", default="fp32", choices=["fp32", "bf16"])
ap.add_argument("--name", default=None)
ap.add_argument("--max-minutes", type=float, default=18)
a = ap.parse_args()
dtype = {"fp32": torch.float32, "bf16": torch.bfloat16}[a.dtype]
ns = [int(x) for x in a.ns.split(",")]
grid = [tuple(int(y) for y in x.split(":")) for x in a.grid.split(",")]
set_seed(0)
run = start_run("kvcache", vars(a), name=f"equiv_{a.dtype}" + (f"_{a.name}" if a.name else ""))
L = load_target(finetuned=True, merge=True, dtype=dtype)
tok, model = L.tokenizer, L.model
rows, t_start = [], time.time()
for P, nmax in grid:
    ids, meta = prompt_of_length(tok, P)
    print(f"P={P} real={ids.shape[1]} nmax={nmax}", flush=True)
    tn, tk = [], []
    rn = greedy_nocache(model, ids, nmax, set(), force_len=True, trace=tn)
    rk = greedy_kv(model, ids, nmax, set(), force_len=True, trace=tk)
    for N in [n for n in ns if n <= nmax]:
        A, B = rn.tokens[:N], rk.tokens[:N]
        div = next((i for i in range(N) if A[i] != B[i]), None)
        last = N - 1 if div is None else div
        dl = [float((tn[i] - tk[i]).abs().max()) for i in range(last + 1)]
        row = {"dtype": a.dtype, "P_target": P, "P_actual": meta["n_tokens"], "N": N, "exact_match": div is None,
               "first_divergence": div, "max_abs_dlogit_upto_div": max(dl), "max_abs_dlogit_step0": dl[0],
               "mean_abs_dlogit": sum(dl) / len(dl)}
        if div is not None:
            t2 = torch.topk(tn[div], 2).values
            k2 = torch.topk(tk[div], 2).values
            row.update(margin_nocache=float(t2[0] - t2[1]), margin_kv=float(k2[0] - k2[1]),
                       tok_nocache=A[div], tok_kv=B[div],
                       top2_nocache=[tok.decode([int(i)]) for i in torch.topk(tn[div], 2).indices])
        rows.append(row)
        print(row, flush=True)
    if (time.time() - t_start) / 60 > a.max_minutes:
        print("time budget reached; stopping grid", flush=True)
        break
keys = sorted({k for r in rows for k in r})
with open(run / "equivalence.csv", "w", newline="") as f:
    w = csv.DictWriter(f, keys)
    w.writeheader()
    w.writerows(rows)
n_ok = sum(r["exact_match"] for r in rows)
finish_run(run, {"rows": rows, "cells": len(rows), "exact": n_ok, "minutes": (time.time() - t_start) / 60},
           headline=f"{a.dtype}: {n_ok}/{len(rows)} cells exact token match")
print("run dir", run)
