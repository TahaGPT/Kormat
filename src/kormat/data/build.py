"""`python -m kormat.data.build`: raw cache -> processed splits, manifest, EDA, token stats, figures."""
from __future__ import annotations

import hashlib
import json
from collections import Counter

import numpy as np

from kormat.common.paths import FIGURES, PROCESSED, RESULTS
from kormat.data.load import ALLOW, load_raw
from kormat.data.sentences import parse_numbered_note
from kormat.data.split import ALGO, SEED, patient_split
from kormat.prompting.labels import LABELS

MAX_LEN = 1024


def make_records(rows: list[dict]) -> tuple[list[dict], dict]:
    recs, bad = [], {"non_contiguous_ids": [], "evidence_not_in_note": [], "bad_label": [], "empty_note": [], "empty_criterion_text": []}
    for r in sorted(rows, key=lambda r: r["annotation_id"]):
        sents = parse_numbered_note(r["note"])
        ev = sorted({int(i) for i in json.loads(r["expert_sentences"])})
        a = r["annotation_id"]
        if not (r["criterion_text"] or "").strip():
            bad["empty_criterion_text"].append(a)
        if not sents:
            bad["empty_note"].append(a)
        if list(sents) != list(range(len(sents))):
            bad["non_contiguous_ids"].append(a)
        if any(i not in sents for i in ev):
            bad["evidence_not_in_note"].append(a)
        if r["expert_eligibility"] not in LABELS[r["criterion_type"]]:
            bad["bad_label"].append(a)
        recs.append({"annotation_id": a, "patient_id": r["patient_id"], "trial_id": r["trial_id"],
                     "trial_title": r["trial_title"], "criterion_type": r["criterion_type"],
                     "criterion_text": r["criterion_text"] or "", "sentences": sents,
                     "label": r["expert_eligibility"], "evidence": ev})
    return recs, bad


def _q(x):
    return {"min": int(np.min(x)), "median": float(np.median(x)), "p95": float(np.percentile(x, 95)),
            "p99": float(np.percentile(x, 99)), "max": int(np.max(x)), "mean": round(float(np.mean(x)), 1)}


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows, rev = load_raw()
    assert set(rows[0]) == set(ALLOW) and len(rows) == 1015
    recs, bad = make_records(rows)
    sp = patient_split([r["patient_id"] for r in recs])
    where = {p: s for s, ps in sp.items() for p in ps}
    by = {s: [r for r in recs if where[r["patient_id"]] == s] for s in sp}
    PROCESSED.mkdir(parents=True, exist_ok=True)
    sha = {}
    for s, rs in by.items():
        txt = "".join(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n" for r in rs)
        (PROCESSED / f"{s}.jsonl").write_text(txt)
        sha[s] = hashlib.sha256(txt.encode()).hexdigest()
    assert sum(len(v) for v in by.values()) == 1015

    lab = {s: {t: dict(Counter(r["label"] for r in rs if r["criterion_type"] == t)) for t in LABELS}
           for s, rs in by.items()}
    manifest = {"seed": SEED, "algorithm": ALGO, "dataset": "ncbi/TrialGPT-Criterion-Annotations",
                "dataset_revision": rev, "patients": sp,
                "n_patients": {s: len(v) for s, v in sp.items()}, "n_annotations": {s: len(v) for s, v in by.items()},
                "n_trials": {s: len({r["trial_id"] for r in rs}) for s, rs in by.items()},
                "n_trials_total": len({r["trial_id"] for r in recs}),
                "label_counts": lab, "sha256": sha}
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "split_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))

    # EDA
    d = RESULTS / "data"
    d.mkdir(exist_ok=True)
    eda = {"n_rows": len(recs), "n_patients": 53, "dropped_columns": [
        "gpt4_explanation", "explanation_correctness", "gpt4_sentences", "gpt4_eligibility", "training"],
        "gold_validation": {k: {"count": len(v), "annotation_ids": v} for k, v in bad.items()},
        "label_by_type": {t: dict(Counter(r["label"] for r in recs if r["criterion_type"] == t)) for t in LABELS},
        "sentences_per_note": _q([len(r["sentences"]) for r in recs]),
        "evidence_len": _q([len(r["evidence"]) for r in recs]),
        "empty_evidence_by_label": dict(Counter(r["label"] for r in recs if not r["evidence"])),
        "evidence_len_distribution": dict(sorted(Counter(len(r["evidence"]) for r in recs).items()))}
    (d / "eda.json").write_text(json.dumps(eda, indent=2, sort_keys=True))

    # label distribution figure
    labs = list(dict.fromkeys(l for t in LABELS for l in LABELS[t]))
    fig, ax = plt.subplots(figsize=(7, 3.4))
    w = 0.38
    for k, t in enumerate(LABELS):
        ax.bar(np.arange(len(labs)) + (k - .5) * w, [eda["label_by_type"][t].get(l, 0) for l in labs], w, label=t)
    ax.set_xticks(range(len(labs)))
    ax.set_xticklabels(labs, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("annotations")
    ax.set_xlabel("expert label")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURES / "label_distribution.pdf")
    plt.close(fig)

    # token stats
    from transformers import AutoTokenizer
    from kormat.data.sft import encode_example
    from kormat.modeling import TARGET_ID
    tok = AutoTokenizer.from_pretrained(TARGET_ID)
    lens, trunc = {}, {}
    for s, rs in by.items():
        rs = [r for r in rs if r["criterion_text"].strip()]  # 1 row has a null criterion; unusable
        lens[s] = [len(encode_example(r, tok, None)["input_ids"]) for r in rs]
        over = [r for r, n in zip(rs, lens[s]) if n > MAX_LEN]
        trunc[s] = {"n_over_max_len": len(over), "annotation_ids": [r["annotation_id"] for r in over]}
    allv = sum(lens.values(), [])
    stats = {"tokenizer": TARGET_ID, "max_len": MAX_LEN, "per_split": {s: _q(v) for s, v in lens.items()},
             "all": _q(allv), "over_max_len": trunc,
             "policy": (f"max_len={MAX_LEN} tokens (full chat = prompt + target + <|im_end|>). Examples above max_len are "
                        "dropped from SFT, never cut: truncating the note would delete sentences and break evidence IDs, "
                        "truncating the end would remove the target. Evaluation uses the full untruncated prompt."),
             "gold_evidence_lost": sum(v["n_over_max_len"] for v in trunc.items() and trunc.values()),
             "note": "no example exceeds max_len, so nothing is truncated or dropped (except the single null-criterion row)"}
    (d / "token_stats.json").write_text(json.dumps(stats, indent=2, sort_keys=True))
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    ax.hist([lens[s] for s in ("train", "val", "test")], bins=30, stacked=True, label=["train", "val", "test"])
    ax.axvline(MAX_LEN, color="k", ls="--", lw=1, label=f"max_len={MAX_LEN}")
    ax.set_xlabel("tokens per example (prompt + target)")
    ax.set_ylabel("examples")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES / "token_lengths.pdf")
    plt.close(fig)
    print(json.dumps({"counts": manifest["n_annotations"], "bad": {k: len(v) for k, v in bad.items()},
                      "tokens": stats["per_split"], "over": {s: v["n_over_max_len"] for s, v in trunc.items()}}, indent=1))


if __name__ == "__main__":
    main()
