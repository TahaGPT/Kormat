"""TGA-35: speculative-decoding benchmark (CPU) + theory curve.

Workloads: A = real task (val prompts, frozen prompt contract, stop at EOS, JSON output);
           B = free-text rationale (same note, plain-prose instruction, force_len tokens).
c = measured ratio of one q_len=1 decode-step time (draft / target) at a realistic cache length.
Theory: expected tokens per iteration E = (1-a^(g+1))/(1-a); speedup = E/(g*c+1)  (Leviathan et al.).
alpha_hat uses the truncated-geometric MLE accepted/(accepted + #iterations that ended in a rejection).
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
ap.add_argument("--workload", default="A", choices=["A", "B"])
ap.add_argument("--n", type=int, default=10)
ap.add_argument("--gammas", default="1,2,3,4,6")
ap.add_argument("--max-new", type=int, default=64)
ap.add_argument("--plot", action="store_true")
ap.add_argument("--measure-c", type=int, default=1)
ap.add_argument("--micro", action="store_true", help="linear-layer M=1 vs M>1 micro-benchmark (no model)")
a = ap.parse_args()


def theory(alpha, g, c, v=1.0):
    """v = cost of the (g+1)-token verify forward in units of one target decode step (paper assumes v=1)."""
    E = (g + 1) if alpha >= 1 else (1 - alpha ** (g + 1)) / (1 - alpha)
    return E / (g * c + v)


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, wl in zip(axs, "AB"):
        fs = sorted(glob.glob(str(RESULTS / "speculative" / "*" / f"summary_{wl}.json")))
        if not fs:
            continue
        s = json.load(open(fs[-1]))
        gs = [r["gamma"] for r in s["by_gamma"]]
        ax.plot(gs, [r["speedup_total"] for r in s["by_gamma"]], "o-", label="measured (total latency)")
        ax.plot(gs, [r["speedup_decode"] for r in s["by_gamma"]], "s--", label="measured (decode only)")
        ax.plot(gs, [theory(r["alpha_hat"], r["gamma"], s["c"]) for r in s["by_gamma"]], "k:", label=f"theory, c={s['c']:.2f}")
        ve = s["meta"].get("verify_ms", {})
        t1 = s["meta"]["step_ms"]["target_q1"]
        if ve:
            ax.plot(gs, [theory(r["alpha_hat"], r["gamma"], s["c"], ve[str(r["gamma"] + 1)] / t1) for r in s["by_gamma"]],
                    "r-.", label="theory with measured verify cost")
        ax.axhline(1, color="grey", lw=.8)
        ax.set_xlabel("gamma"); ax.set_ylabel("speedup vs greedy KV"); ax.set_title(f"workload {wl}"); ax.grid(alpha=.3)
        ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(FIGURES / "spec_speedup.pdf"); fig.savefig(FIGURES / "spec_speedup.png", dpi=150)
    fig, ax = plt.subplots(figsize=(5, 3.4))
    for wl in "AB":
        fs = sorted(glob.glob(str(RESULTS / "speculative" / "*" / f"summary_{wl}.json")))
        if fs:
            s = json.load(open(fs[-1]))
            ax.plot([r["gamma"] for r in s["by_gamma"]], [r["acceptance_rate"] for r in s["by_gamma"]], "o-", label=f"workload {wl}")
    ax.set_xlabel("gamma"); ax.set_ylabel("acceptance rate (accepted/drafted)"); ax.set_ylim(0, 1); ax.grid(alpha=.3); ax.legend()
    fig.tight_layout(); fig.savefig(FIGURES / "spec_acceptance.pdf"); fig.savefig(FIGURES / "spec_acceptance.png", dpi=150)
    print("figures written")


if a.plot:
    plot()
    raise SystemExit

if a.micro:
    out = {}
    torch.manual_seed(0)
    for name, shape in (("mlp_up_8960x1536", (8960, 1536)), ("lm_head_151936x1536", (151936, 1536))):
        w = torch.randn(*shape)
        for m in (1, 2, 3, 5, 9):
            x = torch.randn(m, shape[1])
            ts = []
            for _ in range(12):
                t0 = time.perf_counter(); torch.nn.functional.linear(x, w); ts.append(time.perf_counter() - t0)
            out[f"{name}_M{m}_ms"] = st.median(ts[2:]) * 1e3
    d = RESULTS / "speculative"; d.mkdir(parents=True, exist_ok=True)
    (d / "micro_linear.json").write_text(json.dumps({"loadavg": __import__("os").getloadavg(), **out}, indent=1))
    print(json.dumps(out, indent=1))
    raise SystemExit

import os  # noqa: E402

from kormat.data import load_split  # noqa: E402
from kormat.decoding import greedy_kv, speculative_greedy  # noqa: E402
from kormat.decoding.api import _fwd  # noqa: E402
from kormat.decoding.cache import PreallocKVCache  # noqa: E402
from kormat.decoding.workloads import record_prompt  # noqa: E402
from kormat.modeling import eos_ids, load_draft, load_target  # noqa: E402
from kormat.prompting.template import build_messages, render_prompt  # noqa: E402

set_seed(0)
run = start_run("speculative", vars(a), name=f"bench_{a.workload}")
T, D = load_target(True, True), load_draft()
tok = T.tokenizer
eos = eos_ids(tok, T.model)
val = load_split("val")
sel = val[:: max(1, len(val) // a.n)][: a.n]
meta = {"loadavg_start": os.getloadavg()}


def prompt_ids(rec):
    if a.workload == "A":
        return record_prompt(tok, rec)
    msgs = build_messages(rec["sentences"], rec["criterion_type"], rec["criterion_text"])
    u = msgs[1]["content"]
    msgs[1]["content"] = u[:u.index("Allowed labels:")] + (
        "Write a long, detailed plain-prose explanation (no JSON, no lists) of how the note relates to this criterion, "
        "going through the relevant facts one by one.")
    msgs[0] = {"role": "system", "content": "You are a careful clinical-trial eligibility screener. Answer in plain prose."}
    return tok(render_prompt(tok, msgs), return_tensors="pt", add_special_tokens=False).input_ids


def step_time(model, P, q, reps=15):
    """Median wall time of one forward with q new tokens appended to a cache of length P (decode step / verify block)."""
    ids = torch.randint(100, 5000, (1, P))
    ts = []
    with torch.inference_mode():
        for _ in range(reps + 2):
            c = PreallocKVCache.for_model(model, 1, P + 16)
            _fwd(model, ids, c)
            blk = torch.randint(100, 5000, (1, q))
            t0 = time.perf_counter()
            _fwd(model, blk, c, last_only=False)
            ts.append(time.perf_counter() - t0)
    return st.median(ts[2:])


if a.measure_c:
    Pc = 330
    td, tt = step_time(D.model, Pc, 1), step_time(T.model, Pc, 1)
    meta["c"] = td / tt
    meta["step_ms"] = {"draft_q1": td * 1e3, "target_q1": tt * 1e3}
    meta["verify_ms"] = {q: step_time(T.model, Pc, q, 8) * 1e3 for q in (2, 3, 4, 5, 7, 9)}
    meta["c_P"] = Pc
    print("c =", meta["c"], meta, flush=True)
c = meta.get("c", 0.4)
first = prompt_ids(sel[0])
greedy_kv(T.model, first, 4, eos, force_len=True)      # warm-up
speculative_greedy(T.model, D.model, first, 4, eos, 2, force_len=True)
rows = []
fl = a.workload == "B"
for pi, rec in enumerate(sel):
    ids = prompt_ids(rec)
    base = greedy_kv(T.model, ids, a.max_new, eos, tok, force_len=fl)
    rows.append(dict(prompt=pi, P=ids.shape[1], method="greedy_kv", gamma=0, n_new=base.n_new, prefill_ms=base.prefill_ms,
                     decode_ms=base.decode_ms, total_tok_s=base.tokens_per_s, peak_rss_mb=base.peak_rss_mb,
                     kv_mb=base.kv_cache_mb, exact=True))
    if pi == 0:
        meta["sample_output"] = base.text[:400]
    for g in (int(x) for x in a.gammas.split(",")):
        r = speculative_greedy(T.model, D.model, ids, a.max_new, eos, g, tok, force_len=fl)
        s = r.stats
        rej = sum(1 for i in s["per_iter"] if i["accepted"] < i["drafted"])
        rows.append(dict(prompt=pi, P=ids.shape[1], method="spec", gamma=g, n_new=r.n_new, prefill_ms=r.prefill_ms,
                         decode_ms=r.decode_ms, total_tok_s=r.tokens_per_s, peak_rss_mb=r.peak_rss_mb, kv_mb=r.kv_cache_mb,
                         exact=r.tokens == base.tokens, drafted=s["drafted"], accepted=s["accepted"],
                         iterations=s["iterations"], rejections=rej))
        print(rows[-1], flush=True)
    with open(run / f"bench_{a.workload}.csv", "w", newline="") as f:
        keys = sorted({k for r in rows for k in r})
        w = csv.DictWriter(f, keys); w.writeheader(); w.writerows(rows)

by = []
for g in (int(x) for x in a.gammas.split(",")):
    S = [r for r in rows if r["method"] == "spec" and r["gamma"] == g]
    B = {r["prompt"]: r for r in rows if r["method"] == "greedy_kv"}
    dr, ac, rj = sum(r["drafted"] for r in S), sum(r["accepted"] for r in S), sum(r["rejections"] for r in S)
    it = sum(r["iterations"] for r in S)
    tot = lambda r: r["prefill_ms"] + r["decode_ms"]
    by.append({"gamma": g, "acceptance_rate": ac / dr if dr else 0, "alpha_hat": ac / (ac + rj) if ac + rj else 0,
               "mean_emitted_per_iter": (ac + it) / it, "all_exact": all(r["exact"] for r in S),
               "speedup_total": st.median(tot(B[r["prompt"]]) / tot(r) for r in S),
               "speedup_decode": st.median(B[r["prompt"]]["decode_ms"] / r["decode_ms"] for r in S),
               "tok_s_median": st.median(r["total_tok_s"] for r in S), "peak_rss_mb_median": st.median(r["peak_rss_mb"] for r in S),
               "kv_mb_median": st.median(r["kv_mb"] for r in S)})
base_rows = [r for r in rows if r["method"] == "greedy_kv"]
summary = {"workload": a.workload, "c": c, "meta": meta, "n_prompts": len(sel), "by_gamma": by,
           "baseline": {"tok_s_median": st.median(r["total_tok_s"] for r in base_rows),
                        "peak_rss_mb_median": st.median(r["peak_rss_mb"] for r in base_rows),
                        "kv_mb_median": st.median(r["kv_mb"] for r in base_rows),
                        "n_new_median": st.median(r["n_new"] for r in base_rows)}}
meta["loadavg_end"] = os.getloadavg()
(run / f"summary_{a.workload}.json").write_text(json.dumps(summary, indent=1))
finish_run(run, summary, headline=f"spec bench {a.workload}: c={c:.2f}, " + ", ".join(f"g{b['gamma']}:{b['speedup_total']:.2f}x" for b in by))
print("run dir", run)
