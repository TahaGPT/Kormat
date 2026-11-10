"""Auto-tag errors in a predictions.jsonl (TGA-23 tooling). python -m kormat.eval.errors <predictions.jsonl>"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

from kormat.prompting.labels import INVALID


def tag_row(r: dict) -> list[str]:
    tags = []
    g, p = r["gold_label"], r["parsed_label"]
    gi, pi = set(r["gold_ids"]), set(r["parsed_ids"])
    if p == INVALID:
        tags.append("invalid_output")
    elif g != p:
        tags.append(f"label:{g}->{p}")
        if p == "not enough information":
            tags.append("over_abstain")
        elif g == "not enough information":
            tags.append("unsupported_commitment")
        elif g == "not applicable" or p == "not applicable":
            tags.append("applicability_confusion")
        else:
            tags.append("polarity_error")
    if r.get("hallucinated_ids"):
        tags.append("hallucinated_evidence_id")
    if gi != pi:
        if not pi and gi:
            tags.append("missed_evidence")
        elif pi and not gi:
            tags.append("spurious_evidence")
        elif not (gi & pi):
            tags.append("wrong_evidence")
        else:
            tags.append("partial_evidence")
    if g == p and p != INVALID and pi != gi:
        tags.append("right_label_wrong_evidence")
    return tags


def tag_file(path) -> list[dict]:
    out = []
    for line in Path(path).read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            r["error_tags"] = tag_row(r)
            out.append(r)
    return out


if __name__ == "__main__":
    rows = tag_file(sys.argv[1])
    errs = [r for r in rows if r["error_tags"]]
    print(json.dumps(Counter(t for r in errs for t in r["error_tags"]).most_common(), indent=1))
    dest = Path(sys.argv[1]).with_name("errors_tagged.jsonl")
    dest.write_text("".join(json.dumps(r) + "\n" for r in errs))
    print(f"{len(errs)}/{len(rows)} rows with errors -> {dest}")
