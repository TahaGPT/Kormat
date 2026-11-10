"""Classification metrics with fixed label lists; INVALID is a predicted-only column.

Macro-F1 convention: unweighted mean of per-class F1 over the classes of the section's label set that occur
in the gold labels or the predictions (a class absent from both is undefined and not counted as 0). The
fixed-list variant (absent classes scored 0) is reported as `macro_f1_fixed` for transparency.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

from kormat.prompting.labels import ALL_LABELS, INVALID, LABELS


def _present(gold, pred, labels):
    seen = set(gold) | set(pred)
    return [l for l in labels if l in seen] or list(labels)


def _section(gold, pred, labels):
    if not gold:
        return {"n": 0, "accuracy": None, "macro_f1": None, "weighted_f1": None, "per_class": {},
                "confusion": {"labels": list(labels) + [INVALID], "matrix": []}}
    p, r, f, s = precision_recall_fscore_support(gold, pred, labels=list(labels), zero_division=0)
    cols = list(labels) + [INVALID]
    idx = {l: i for i, l in enumerate(cols)}
    m = np.zeros((len(labels), len(cols)), dtype=int)
    for g, q in zip(gold, pred):
        if g in labels:
            m[labels.index(g), idx.get(q, len(cols) - 1)] += 1
    return {
        "n": len(gold),
        "accuracy": float(accuracy_score(gold, pred)),
        "macro_f1": float(f1_score(gold, pred, average="macro", labels=_present(gold, pred, labels), zero_division=0)),
        "macro_f1_fixed": float(f1_score(gold, pred, average="macro", labels=list(labels), zero_division=0)),
        "macro_labels": _present(gold, pred, labels),
        "weighted_f1": float(f1_score(gold, pred, average="weighted", labels=list(labels), zero_division=0)),
        "per_class": {l: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
                      for i, l in enumerate(labels)},
        "confusion": {"labels": cols, "matrix": m.tolist()},
    }


def classification_report(gold, pred, types) -> dict:
    gold, pred, types = list(gold), list(pred), list(types)
    rep = {"overall": _section(gold, pred, list(ALL_LABELS))}
    for t in ("inclusion", "exclusion"):
        sel = [i for i, x in enumerate(types) if x == t]
        rep[t] = _section([gold[i] for i in sel], [pred[i] for i in sel], list(LABELS[t]))
    return rep


def bootstrap_ci(gold, pred, metric="macro_f1", n=1000, seed=42, labels=None, alpha=0.05):
    """Percentile bootstrap CI over examples. metric: 'accuracy' | 'macro_f1' | callable(g,p)."""
    gold, pred = np.asarray(list(gold), dtype=object), np.asarray(list(pred), dtype=object)
    labels = list(labels or ALL_LABELS)
    if callable(metric):
        fn = metric
    elif metric == "accuracy":
        fn = lambda g, p: float(np.mean(g == p))
    else:
        fn = lambda g, p: float(f1_score(g, p, average="macro", labels=_present(list(g), list(p), labels),
                                         zero_division=0))
    rng = np.random.default_rng(seed)
    N = len(gold)
    vals = []
    for _ in range(n):
        ix = rng.integers(0, N, N)
        vals.append(fn(gold[ix], pred[ix]))
    return {"point": fn(gold, pred), "lo": float(np.percentile(vals, 100 * alpha / 2)),
            "hi": float(np.percentile(vals, 100 * (1 - alpha / 2))), "n_boot": n, "seed": seed}


def plot_confusion(section, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels, M = section["confusion"]["labels"], np.array(section["confusion"]["matrix"])
    rows = [l for l in labels if l != INVALID][: M.shape[0]]
    rn = M / np.maximum(M.sum(1, keepdims=True), 1) * 100
    fig, ax = plt.subplots(figsize=(1.3 * len(labels) + 2, 1.1 * len(rows) + 2))
    ax.imshow(rn, cmap="Blues", vmin=0, vmax=100)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]}\n{rn[i, j]:.0f}%", ha="center", va="center",
                    color="white" if rn[i, j] > 55 else "black", fontsize=8)
    ax.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
    ax.set_yticks(range(len(rows)), rows)
    ax.set_xlabel("predicted")
    ax.set_ylabel("gold")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
