from kormat.common.runlog import capture_env
from kormat.prompting.template import build_messages, format_target


def test_env_capture():
    assert "python" in capture_env()


def test_prompt_contract_shape():
    msgs = build_messages({0: "A 58-year-old woman.", 1: "She smokes."}, "exclusion", " Current smoker ")
    assert msgs[0]["role"] == "system" and msgs[1]["role"] == "user"
    assert "0. A 58-year-old woman.\n1. She smokes.\n\nCriterion type: exclusion\nCriterion: Current smoker\n" in msgs[1]["content"]
    assert "Allowed labels: excluded | not excluded | not enough information | not applicable" in msgs[1]["content"]
    assert format_target("excluded", [1, 1, 0]) == '{"label": "excluded", "evidence": [0, 1]}'
