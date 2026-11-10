"""Evidence-sentence metrics with explicit empty-set rules."""
from __future__ import annotations


def prf_pair(G: set, P: set):
    """Per-record (P,R,F1). Empty rules: G=P=empty -> 1,1,1; G empty,P not -> 0,1,0; G not,P empty -> 1,0,0."""
    if not G and not P:
        return 1.0, 1.0, 1.0
    if not G:
        return 0.0, 1.0, 0.0
    if not P:
        return 1.0, 0.0, 0.0
    tp = len(G & P)
    p, r = tp / len(P), tp / len(G)
    return p, r, (2 * p * r / (p + r) if tp else 0.0)


def _prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def _block(G, P):
    tp = sum(len(g & p) for g, p in zip(G, P))
    fp = sum(len(p - g) for g, p in zip(G, P))
    fn = sum(len(g - p) for g, p in zip(G, P))
    mp, mr, mf = _prf(tp, fp, fn)
    pairs = [prf_pair(g, p) for g, p in zip(G, P)]
    n = len(pairs) or 1
    ne = [x for x, g in zip(pairs, G) if g]
    m = len(ne) or 1
    cells = {"gold_empty_pred_empty": 0, "gold_empty_pred_nonempty": 0,
             "gold_nonempty_pred_empty": 0, "gold_nonempty_pred_nonempty": 0}
    for g, p in zip(G, P):
        cells[f"gold_{'non' if g else ''}empty_pred_{'non' if p else ''}empty".replace("nonempty_pred", "nonempty_pred")] += 1
    return {
        "n": len(G), "tp": tp, "fp": fp, "fn": fn,
        "micro": {"precision": mp, "recall": mr, "f1": mf},
        "macro_all": {"precision": sum(x[0] for x in pairs) / n, "recall": sum(x[1] for x in pairs) / n,
                      "f1": sum(x[2] for x in pairs) / n},
        "macro_nonempty_gold": {"precision": sum(x[0] for x in ne) / m, "recall": sum(x[1] for x in ne) / m,
                                "f1": sum(x[2] for x in ne) / m, "n": len(ne)},
        "cells": cells,
    }


def evidence_report(gold_ids_list, pred_ids_list, types, labels) -> dict:
    G = [set(g) for g in gold_ids_list]
    P = [set(p) for p in pred_ids_list]
    types, labels = list(types), list(labels)
    rep = _block(G, P)
    for key, vals in (("by_type", types), ("by_gold_label", labels)):
        rep[key] = {}
        for v in sorted(set(vals)):
            ix = [i for i, x in enumerate(vals) if x == v]
            rep[key][v] = _block([G[i] for i in ix], [P[i] for i in ix])
    return rep
