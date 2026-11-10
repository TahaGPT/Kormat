"""Golden test: the frozen prompt contract must not drift."""
import hashlib
import json

from kormat.prompting.template import SYSTEM_PROMPT, build_messages, format_target

SENT = {0: "A 58-year-old woman has chest pain.", 1: "She takes aspirin."}
GOLDEN_USER = (
    "Patient note:\n0. A 58-year-old woman has chest pain.\n1. She takes aspirin.\n\n"
    "Criterion type: inclusion\nCriterion: Chronic stable angina.\n"
    "Allowed labels: included | not included | not enough information | not applicable\n"
    'Return: {"label": "<allowed label>", "evidence": [<sentence ids>]}'
)


def test_user_message_golden():
    m = build_messages(SENT, "inclusion", "  Chronic stable angina. ")
    assert [x["role"] for x in m] == ["system", "user"]
    assert m[1]["content"] == GOLDEN_USER


def test_system_prompt_hash():
    assert hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest()[:16] == "589635efa4f7d074"


def test_exclusion_labels():
    assert "excluded | not excluded | not enough information | not applicable" in build_messages(SENT, "exclusion", "x")[1]["content"]


def test_target_format_golden():
    assert format_target("included", [2, 0, 2]) == '{"label": "included", "evidence": [0, 2]}'
    assert format_target("not applicable", []) == '{"label": "not applicable", "evidence": []}'
    assert json.loads(format_target("included", [1]))["evidence"] == [1]


def test_bad_type():
    import pytest
    with pytest.raises(ValueError):
        build_messages(SENT, "other", "x")
