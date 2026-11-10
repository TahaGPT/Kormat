"""Token-for-token equivalence + speedup check: prefix-cache generation vs plain generate."""
from __future__ import annotations

import json
import sys
from pathlib import Path

from kormat.eval.run import PrefixCache, _rec, generate_one
from kormat.modeling import eos_ids


def check(loaded, records, out_path=None) -> dict:
    recs = [_rec(r) for r in records]
    order = {}
    for r in recs:
        order.setdefault(r.get("patient_id"), len(order))
    recs.sort(key=lambda r: order[r.get("patient_id")])
    eos = eos_ids(loaded.tokenizer, loaded.model)
    pc, rows = PrefixCache(), []
    for r in recs:
        a = generate_one(loaded.model, loaded.tokenizer, eos, r, None, return_ids=True)
        b = generate_one(loaded.model, loaded.tokenizer, eos, r, pc, return_ids=True)
        rows.append({"annotation_id": r["annotation_id"], "match": a["new_ids"] == b["new_ids"],
                     "n_new": a["n_new_tokens"], "plain_ms": a["latency_ms"], "cached_ms": b["latency_ms"],
                     "prefix_len": b.get("prefix_len"), "suffix_len": b.get("suffix_len"),
                     "prefix_reused": b.get("prefix_reused")})
    res = {"n": len(rows), "all_match": all(x["match"] for x in rows),
           "max_new_tokens": max((x["n_new"] for x in rows), default=0),
           "mean_plain_ms": sum(x["plain_ms"] for x in rows) / max(len(rows), 1),
           "mean_cached_ms": sum(x["cached_ms"] for x in rows) / max(len(rows), 1),
           "fallbacks": pc.fallbacks, "reused": pc.reused, "rows": rows}
    res["speedup"] = res["mean_plain_ms"] / res["mean_cached_ms"] if res["mean_cached_ms"] else None
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(res, indent=2))
    return res


if __name__ == "__main__":
    from kormat.data import load_split
    from kormat.modeling import load_target, set_threads

    set_threads()
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    kind = sys.argv[2] if len(sys.argv) > 2 else "ft"
    # take rows from a few patients so that prefix reuse occurs
    val = [_rec(r) for r in load_split("val")]
    first = val[0]["patient_id"]
    same = [r for r in val if r["patient_id"] == first]
    recs = (same + [r for r in val if r["patient_id"] != first])[:n]
    out = check(load_target(finetuned=(kind == "ft"), merge=True), recs, "results/eval/prefix_cache_check.json")
    print({k: v for k, v in out.items() if k != "rows"})
