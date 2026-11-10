"""Deterministic explanations built only from cited sentences."""
from __future__ import annotations

from kormat.prompting.labels import INVALID


def build_explanation(label, evidence_ids, sentences, criterion_type, criterion_text) -> str:
    if label == INVALID:
        return "The model output could not be parsed into a valid label, so no judgement is made for this criterion."
    ids = [i for i in sorted(set(evidence_ids)) if i in sentences]
    if label == "not applicable" and not ids:
        return f"No sentence is cited; the {criterion_type} criterion is judged 'not applicable' to this patient."
    if not ids:
        if label == "not enough information":
            return ("No sentence in the note addresses this criterion, so the information is insufficient; "
                    "absence of evidence is not treated as proof either way.")
        return f"No supporting sentence is cited. The {criterion_type} criterion is judged '{label}' without cited evidence."
    parts = [f"Sentence {i} states: '{sentences[i].strip()}'." for i in ids]
    return " ".join(parts) + f" On that basis the {criterion_type} criterion is judged '{label}'."
