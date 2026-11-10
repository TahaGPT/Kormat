"""SFT example builder: completion-only loss (labels=-100 on the prompt)."""
from __future__ import annotations

from kormat.data import load_split
from kormat.prompting.template import build_messages, format_target, render_prompt

IGNORE = -100
END = "<|im_end|>"


def encode_example(rec: dict, tokenizer, max_len: int | None = 1024) -> dict | None:
    """Return one example, or None when longer than max_len (policy: drop, never cut tokens)."""
    if not rec["criterion_text"].strip():
        return None  # unusable row (null criterion in source)
    prompt = render_prompt(tokenizer, build_messages(rec["sentences"], rec["criterion_type"], rec["criterion_text"]))
    full = prompt + format_target(rec["label"], rec["evidence"]) + END
    p_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    ids = tokenizer(full, add_special_tokens=False)["input_ids"]
    if ids[: len(p_ids)] != p_ids:
        raise AssertionError(f"prompt is not an exact token prefix (annotation {rec['annotation_id']})")
    if max_len is not None and len(ids) > max_len:
        return None
    labels = [IGNORE] * len(p_ids) + ids[len(p_ids):]
    return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels}


def build_sft(split: str, tokenizer, max_len: int = 1024) -> list[dict]:
    out = [encode_example(r, tokenizer, max_len) for r in load_split(split)]
    return [e for e in out if e is not None]
