"""Robust parsing of model output into (label, evidence). Never maps garbage to a real class."""
from __future__ import annotations

import json
import re

from kormat.prompting.labels import INVALID, LABELS

SYNONYMS = {
    "insufficient information": "not enough information",
    "insufficient": "not enough information",
    "not enough info": "not enough information",
    "unknown": "not enough information",
    "n/a": "not applicable",
    "na": "not applicable",
    "not relevant": "not applicable",
    "met": "included",
    "not met": "not included",
    "not_included": "not included",
    "not_excluded": "not excluded",
    "not_enough_information": "not enough information",
    "not_applicable": "not applicable",
}
_THINK = re.compile(r"<think>.*?(</think>|$)", re.S | re.I)
_FENCE = re.compile(r"```(?:json|JSON)?", re.I)


def normalise_label(s) -> str | None:
    if not isinstance(s, str):
        return None
    t = re.sub(r"\s+", " ", s.strip().strip("'\"").strip().lower().rstrip("."))
    t = SYNONYMS.get(t, t)
    return t


def _ids(v, valid: set[int]):
    if isinstance(v, (int, str)) and not isinstance(v, bool):
        v = [v]
    if not isinstance(v, (list, tuple)):
        return [], []
    got = []
    for x in v:
        try:
            if isinstance(x, bool):
                continue
            got.append(int(str(x).strip()) if not isinstance(x, int) else x)
        except (ValueError, TypeError):
            continue
    ok = sorted({i for i in got if i in valid})
    bad = sorted({i for i in got if i not in valid})
    return ok, bad


def _first_json(text: str):
    """First decodable {...} object in text."""
    dec = json.JSONDecoder()
    for m in re.finditer(r"\{", text):
        try:
            obj, _ = dec.raw_decode(text[m.start():])
            if isinstance(obj, dict):
                return obj
        except ValueError:
            continue
    return None


def _lenient(text: str):
    lm = re.search(r"""["']?label["']?\s*[:=]\s*["']([^"']+)["']""", text, re.I)
    if not lm:
        return None
    em = re.search(r"""["']?evidence["']?\s*[:=]\s*\[([^\]]*)\]""", text, re.I)
    ev = re.findall(r"-?\d+", em.group(1)) if em else []
    return {"label": lm.group(1), "evidence": [int(x) for x in ev]}


def parse_output(raw: str, criterion_type: str, valid_ids: set[int]) -> dict:
    out = {"label": INVALID, "evidence": [], "status": "invalid", "hallucinated_ids": []}
    if not isinstance(raw, str) or criterion_type not in LABELS:
        return out
    text = _FENCE.sub("", _THINK.sub("", raw)).replace("```", "").strip()
    obj, status = _first_json(text), "ok"
    if obj is None:
        # single quotes -> double quotes
        obj = _first_json(text.replace("'", '"'))
        status = "lenient"
    if obj is None:
        obj, status = _lenient(text), "lenient"
    if obj is None or "label" not in obj:
        return out
    lab = normalise_label(obj.get("label"))
    if lab is None or lab not in LABELS[criterion_type]:
        return out
    if status == "ok" and lab != str(obj.get("label")):
        status = "lenient"
    ev, bad = _ids(obj.get("evidence", []), valid_ids)
    return {"label": lab, "evidence": ev, "status": status, "hallucinated_ids": bad}
