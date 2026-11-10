"""Preliminary-outcome aggregation rule (ADR-004). A screening aid, never ground truth."""
from __future__ import annotations

LIKELY_INELIGIBLE = "Likely ineligible"
UNDETERMINED = "Undetermined, information missing"
POTENTIAL = "Potentially eligible"

RULE_TEXT = (
    "1. Any exclusion criterion labelled \"excluded\" -> Likely ineligible. "
    "2. Else, any inclusion criterion labelled \"not included\" -> Likely ineligible. "
    "3. Else, any criterion labelled \"not enough information\" (or an unparseable/INVALID answer) "
    "-> Undetermined, information missing. "
    "4. Else -> Potentially eligible. "
    "\"not applicable\" criteria are ignored in the decision but still listed. "
    "This is a preliminary screening aid produced by a model; it is not ground truth and not an enrollment decision."
)


def _t(r):
    return r.get("criterion_type") or r["type"]


def aggregate(results: list[dict]) -> dict:
    """results: dicts with criterion_type, label (may be 'INVALID'), text/id optional."""
    triggers: list[dict] = []
    missing = [r for r in results if r["label"] in ("not enough information", "INVALID")]
    ignored = [r for r in results if r["label"] == "not applicable"]
    excl = [r for r in results if _t(r) == "exclusion" and r["label"] == "excluded"]
    notinc = [r for r in results if _t(r) == "inclusion" and r["label"] == "not included"]
    if excl:
        outcome, why, triggers = LIKELY_INELIGIBLE, "At least one exclusion criterion is met (rule 1).", excl
    elif notinc:
        outcome, why, triggers = LIKELY_INELIGIBLE, "At least one inclusion criterion is not met (rule 2).", notinc
    elif missing:
        outcome, why, triggers = UNDETERMINED, "No disqualifying criterion was found, but some criteria lack information (rule 3).", missing
    else:
        outcome, why = POTENTIAL, "No exclusion is met, every applicable inclusion is met, nothing is missing (rule 4)."
    return {
        "outcome": outcome,
        "reason": why,
        "rule": RULE_TEXT,
        "triggering": [{"index": r.get("index"), "text": r.get("text"), "label": r["label"]} for r in triggers],
        "missing_information": [{"index": r.get("index"), "text": r.get("text"), "type": _t(r)} for r in missing],
        "ignored_not_applicable": [{"index": r.get("index"), "text": r.get("text")} for r in ignored],
        "counts": {k: sum(1 for r in results if r["label"] == k) for k in sorted({r["label"] for r in results})},
    }
