import json

import pytest
from transformers import AutoTokenizer

from kormat.eval.run import main, run_eval
from kormat.modeling import Loaded, tiny_qwen


def _recs():
    return [{"annotation_id": f"a{i}", "patient_id": "p", "criterion_type": "inclusion", "criterion_text": "age>18",
             "sentences": {0: "A 40 year old.", 1: "Has asthma."}, "label": "included", "evidence": [0]} for i in range(3)]


def test_test_split_sealed(monkeypatch):
    monkeypatch.delenv("KORMAT_ALLOW_TEST", raising=False)
    with pytest.raises(SystemExit):
        main(["--model", "ft", "--split", "test"])


@pytest.mark.slow
def test_end_to_end_resume(tmp_path):
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    import torch
    m = tiny_qwen(vocab=len(tok))
    ld = Loaded(m, tok, "tiny", "tiny", None, torch.float32)
    out = tmp_path / "o"
    r1 = run_eval(_recs()[:2], ld, out, "val", "tiny", n_boot=10)
    assert r1["n"] == 2
    lines1 = (out / "predictions.jsonl").read_text().splitlines()
    r2 = run_eval(_recs(), ld, out, "val", "tiny", n_boot=10)
    lines2 = (out / "predictions.jsonl").read_text().splitlines()
    assert r2["n"] == 3 and len(lines2) == 3 and lines2[:2] == lines1
    row = json.loads(lines2[0])
    for k in ["annotation_id", "gold_label", "raw_output", "parsed_label", "parse_status", "explanation",
              "latency_ms", "n_prompt_tokens", "n_new_tokens", "hallucinated_ids", "evidence_text"]:
        assert k in row
    assert (out / "metrics.json").exists() and (out / "confusion_overall.pdf").exists()
