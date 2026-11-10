import pytest

from kormat.data import load_split
from kormat.data.sft import END, IGNORE, build_sft, encode_example
from kormat.modeling import TARGET_ID
from kormat.prompting.template import build_messages, format_target, render_prompt


@pytest.fixture(scope="module")
def tok():
    from transformers import AutoTokenizer
    try:
        return AutoTokenizer.from_pretrained(TARGET_ID, local_files_only=True)
    except Exception:
        pytest.skip("tokenizer not cached")


def test_mask_and_prefix(tok):
    recs = [r for r in load_split("val")[:20] if r["criterion_text"].strip()]
    for r in recs:
        ex = encode_example(r, tok, 1024)
        prompt = render_prompt(tok, build_messages(r["sentences"], r["criterion_type"], r["criterion_text"]))
        n = len(tok(prompt, add_special_tokens=False)["input_ids"])
        assert ex["input_ids"][:n] == tok(prompt, add_special_tokens=False)["input_ids"]
        assert all(x == IGNORE for x in ex["labels"][:n])
        assert ex["labels"][n:] == ex["input_ids"][n:]
        tgt = tok.decode(ex["labels"][n:])
        assert tgt == format_target(r["label"], r["evidence"]) + END
        assert len(ex["attention_mask"]) == len(ex["input_ids"])


def test_max_len_drops(tok):
    assert build_sft("val", tok, max_len=10) == []
    assert len(build_sft("val", tok)) == 206
