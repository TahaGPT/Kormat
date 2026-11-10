"""The single prompt contract used by evaluation, benchmarks and the app.

It matches, byte for byte, the prompt the shared LoRA adapter was trained with
(joint training run, disclosed in README / ADR-003). Do not change any string here
without re-checking adapter behaviour: the golden test in tests/test_template.py guards it.
"""
from __future__ import annotations

import json

from kormat.prompting.labels import LABELS

SYSTEM_PROMPT = (
    "You are a clinical-trial eligibility screener. Judge ONE criterion against a patient note "
    "whose sentences are numbered. Use only facts stated in the note. If the note does not contain "
    "the facts needed, answer \"not enough information\" - missing evidence is never proof. "
    "Answer \"not applicable\" only when the criterion cannot apply to this patient. "
    "Cite the IDs of the sentences that support your label (empty list if none). "
    "Reply with JSON only."
)


def format_note(sentences: dict[int, str]) -> str:
    return "\n".join(f"{i}. {t}" for i, t in sorted(sentences.items()))


def build_messages(sentences: dict[int, str], criterion_type: str, criterion_text: str) -> list[dict]:
    if criterion_type not in LABELS:
        raise ValueError(f"criterion_type must be one of {list(LABELS)}, got {criterion_type!r}")
    allowed = " | ".join(LABELS[criterion_type])
    user = (
        f"Patient note:\n{format_note(sentences)}\n\n"
        f"Criterion type: {criterion_type}\n"
        f"Criterion: {criterion_text.strip()}\n"
        f"Allowed labels: {allowed}\n"
        'Return: {"label": "<allowed label>", "evidence": [<sentence ids>]}'
    )
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def format_target(label: str, evidence: list[int]) -> str:
    return json.dumps({"label": label, "evidence": sorted(set(int(i) for i in evidence))}, separators=(", ", ": "))


def render_prompt(tokenizer, messages: list[dict]) -> str:
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
