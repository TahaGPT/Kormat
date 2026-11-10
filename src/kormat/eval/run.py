"""End-to-end evaluation runner (resumable). python -m kormat.eval.run --model base|ft|both --split val|test"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from kormat.common.paths import RESULTS
from kormat.common import runlog
from kormat.eval.classification import bootstrap_ci, classification_report, plot_confusion
from kormat.eval.evidence import evidence_report
from kormat.eval.explain import build_explanation
from kormat.prompting.labels import INVALID
from kormat.prompting.parse import parse_output
from kormat.prompting.template import build_messages, render_prompt

MAX_NEW = 48


def _rec(r) -> dict:
    if isinstance(r, dict):
        d = dict(r)
    elif hasattr(r, "model_dump"):
        d = r.model_dump()
    elif hasattr(r, "__dict__"):
        d = dict(vars(r))
    else:
        raise TypeError(type(r))
    d["sentences"] = {int(k): v for k, v in d["sentences"].items()}
    return d


class PrefixCache:
    """Holds the KV cache of the shared per-patient prefix (system msg + note) for one patient at a time."""

    def __init__(self):
        self.patient = None
        self.ids = None
        self.cache = None
        self.prefill_ms = 0.0
        self.fallbacks = 0
        self.reused = 0


def _prefix_text(prompt: str) -> str:
    i = prompt.index("\n\nCriterion type:")
    return prompt[: i + 2]


def generate_one(model, tok, eos, rec, pc: PrefixCache | None = None, return_ids=False) -> dict:
    import copy

    import torch

    prompt = render_prompt(tok, build_messages(rec["sentences"], rec["criterion_type"], rec["criterion_text"]))
    enc = tok(prompt, return_tensors="pt")
    ids = enc["input_ids"]
    kw = dict(do_sample=False, max_new_tokens=MAX_NEW, eos_token_id=list(eos),
              pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else min(eos))
    extra = {}
    t0 = time.perf_counter()
    if pc is not None:
        try:
            pre = tok(_prefix_text(prompt), return_tensors="pt")["input_ids"]
            pl = pre.shape[1]
            ok = pl < ids.shape[1] and bool((ids[0, :pl] == pre[0]).all())
        except ValueError:
            ok = False
        if not ok:
            pc.fallbacks += 1
            extra = {"prefix_len": 0, "suffix_len": int(ids.shape[1]), "prefix_reused": False}
            cache = None
        else:
            from transformers import DynamicCache

            if pc.patient != rec.get("patient_id") or pc.ids is None or pc.ids.shape != pre.shape \
                    or not bool((pc.ids == pre).all()):
                with torch.no_grad():
                    c = DynamicCache()
                    model(pre, past_key_values=c, use_cache=True)
                pc.patient, pc.ids, pc.cache, reused = rec.get("patient_id"), pre, c, False
            else:
                reused = True
                pc.reused += 1
            cache = copy.deepcopy(pc.cache)
            extra = {"prefix_len": int(pl), "suffix_len": int(ids.shape[1] - pl), "prefix_reused": reused}
        if cache is not None:
            kw["past_key_values"] = cache
    with torch.no_grad():
        out = model.generate(input_ids=ids, attention_mask=enc["attention_mask"], **kw)
    ms = (time.perf_counter() - t0) * 1000
    new = out[0, ids.shape[1]:]
    res = {"raw_output": tok.decode(new, skip_special_tokens=True), "latency_ms": ms,
           "n_prompt_tokens": int(ids.shape[1]), "n_new_tokens": int(new.shape[0]), **extra}
    if return_ids:
        res["new_ids"] = new.tolist()
    return res


def _row(rec, gen) -> dict:
    p = parse_output(gen["raw_output"], rec["criterion_type"], set(rec["sentences"]))
    ev = {i: rec["sentences"][i] for i in p["evidence"]}
    return {
        "annotation_id": rec["annotation_id"], "patient_id": rec.get("patient_id"),
        "criterion_type": rec["criterion_type"], "criterion_text": rec["criterion_text"],
        "gold_label": rec["label"], "gold_ids": sorted(rec["evidence"]),
        "raw_output": gen["raw_output"], "parsed_label": p["label"], "parsed_ids": p["evidence"],
        "parse_status": p["status"], "hallucinated_ids": p["hallucinated_ids"], "evidence_text": ev,
        "explanation": build_explanation(p["label"], p["evidence"], rec["sentences"], rec["criterion_type"],
                                         rec["criterion_text"]),
        "latency_ms": gen["latency_ms"], "n_prompt_tokens": gen["n_prompt_tokens"],
        "n_new_tokens": gen["n_new_tokens"],
        **{k: gen[k] for k in ("prefix_len", "suffix_len", "prefix_reused") if k in gen},
    }


def compute_metrics(rows: list[dict], n_boot: int = 1000) -> dict:
    gold = [r["gold_label"] for r in rows]
    pred = [r["parsed_label"] for r in rows]
    types = [r["criterion_type"] for r in rows]
    n = max(len(rows), 1)
    m = {"n": len(rows), "classification": classification_report(gold, pred, types),
         "evidence": evidence_report([r["gold_ids"] for r in rows], [r["parsed_ids"] for r in rows], types, gold),
         "invalid_rate": sum(p == INVALID for p in pred) / n,
         "lenient_rate": sum(r["parse_status"] == "lenient" for r in rows) / n,
         "hallucinated_id_rate": sum(bool(r["hallucinated_ids"]) for r in rows) / n,
         "mean_latency_ms": sum(r["latency_ms"] for r in rows) / n}
    if rows:
        m["ci"] = {"accuracy": bootstrap_ci(gold, pred, "accuracy", n=n_boot),
                   "macro_f1": bootstrap_ci(gold, pred, "macro_f1", n=n_boot)}
    return m


def run_eval(records, loaded, out_dir: Path, split: str, name: str, adapter=None, limit=None, n_boot=1000,
             prefix_cache=False, chunk=None):
    """loaded: object with .model and .tokenizer. Resumable via predictions.jsonl."""
    from kormat.modeling import eos_ids

    out_dir.mkdir(parents=True, exist_ok=True)
    recs = [_rec(r) for r in records][: limit or None]
    if prefix_cache:  # group by patient so one prefix prefill serves all its criteria
        order = {}
        for r in recs:
            order.setdefault(r.get("patient_id"), len(order))
        recs.sort(key=lambda r: order[r.get("patient_id")])
    pc = PrefixCache() if prefix_cache else None
    pred_path = out_dir / "predictions.jsonl"
    rows, done = [], set()
    if pred_path.exists():
        for line in pred_path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                rows.append(r)
                done.add(r["annotation_id"])
    cfg = {"name": name, "split": split, "limit": limit, "adapter": adapter,
           "model_id": getattr(loaded, "model_id", None),
           "prefix_cache": prefix_cache, "chunk": chunk,
           "decoding": {"do_sample": False, "max_new_tokens": MAX_NEW, "batch_size": 1,
                        "dtype": str(getattr(loaded, "dtype", "float32"))}}
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2, default=str))
    (out_dir / "env.json").write_text(json.dumps(runlog.capture_env(), indent=2))
    eos = eos_ids(loaded.tokenizer, loaded.model)
    with pred_path.open("a", encoding="utf-8") as f:
        n_new = 0
        for rec in recs:
            if rec["annotation_id"] in done:
                continue
            if chunk is not None and n_new >= chunk:
                break
            n_new += 1
            row = _row(rec, generate_one(loaded.model, loaded.tokenizer, eos, rec, pc))
            f.write(json.dumps(row) + "\n")
            f.flush()
            rows.append(row)
    wanted = {r["annotation_id"] for r in recs}
    rows = [r for r in rows if r["annotation_id"] in wanted]
    metrics = compute_metrics(rows, n_boot)
    metrics["n_requested"] = len(recs)
    metrics["complete"] = len(rows) >= len(recs)
    if pc is not None:
        metrics["prefix_cache"] = {"reused": pc.reused, "fallbacks": pc.fallbacks}
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    if rows:
        for sec in ("overall", "inclusion", "exclusion"):
            if metrics["classification"][sec]["n"]:
                plot_confusion(metrics["classification"][sec], out_dir / f"confusion_{sec}.pdf")
    return metrics


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["base", "ft", "both"], required=True)
    ap.add_argument("--split", choices=["val", "test"], required=True)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--out")
    ap.add_argument("--prefix-cache", action="store_true", help="reuse per-patient prefix KV cache")
    ap.add_argument("--chunk", type=int, help="process at most N new records, then write metrics and exit")
    a = ap.parse_args(argv)
    if a.split == "test" and os.environ.get("KORMAT_ALLOW_TEST") != "1":
        raise SystemExit("Refusing --split test: sealed. Set KORMAT_ALLOW_TEST=1 (lead only).")
    from kormat.data import load_split
    from kormat.modeling import load_target, set_threads

    set_threads()
    records = list(load_split(a.split))
    kinds = ["base", "ft"] if a.model == "both" else [a.model]
    base_out = Path(a.out) if a.out else RESULTS / "eval" / f"{a.split}_{time.strftime('%Y%m%d-%H%M%S')}"
    for k in kinds:
        loaded = load_target(finetuned=(k == "ft"), merge=True)
        out = base_out / k if a.model == "both" else base_out
        m = run_eval(records, loaded, out, a.split, k, adapter=getattr(loaded, "adapter", None), limit=a.limit,
                     prefix_cache=a.prefix_cache, chunk=a.chunk)
        print(k, "n", m["n"], "acc", m["classification"]["overall"]["accuracy"], "macroF1",
              m["classification"]["overall"]["macro_f1"], "invalid", m["invalid_rate"],
              "lat_ms", round(m["mean_latency_ms"], 1))
        del loaded


if __name__ == "__main__":
    main()
