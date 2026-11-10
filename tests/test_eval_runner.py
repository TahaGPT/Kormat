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


@pytest.mark.slow
def test_prefix_cache_equivalence_tiny():
    import torch
    from kormat.eval.prefix_check import check
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    ld = Loaded(tiny_qwen(vocab=len(tok)), tok, "tiny", "tiny", None, torch.float32)
    recs = _recs() + [dict(_recs()[0], annotation_id="b0", patient_id="q", criterion_text="other crit")]
    res = check(ld, recs)
    assert res["all_match"] and res["fallbacks"] == 0 and res["reused"] == 2 and res["n"] == 4


@pytest.mark.slow
def test_chunk(tmp_path):
    import torch
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    ld = Loaded(tiny_qwen(vocab=len(tok)), tok, "tiny", "tiny", None, torch.float32)
    m = run_eval(_recs(), ld, tmp_path, "val", "t", n_boot=5, chunk=2, prefix_cache=True)
    assert m["n"] == 2 and not m["complete"]
    m = run_eval(_recs(), ld, tmp_path, "val", "t", n_boot=5, chunk=2, prefix_cache=True)
    assert m["n"] == 3 and m["complete"]
