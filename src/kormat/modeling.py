"""Shared model loading for eval, benchmarks and the app (CPU-first).

Target = Qwen2.5-1.5B-Instruct (+ shared LoRA adapter), draft = Qwen2.5-0.5B-Instruct.
Real models must be loaded under the model lock (see Makefile / flock .model.lock).
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from kormat.common.paths import CHECKPOINTS

TARGET_ID = "Qwen/Qwen2.5-1.5B-Instruct"
DRAFT_ID = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER_DIR = CHECKPOINTS / "adapter"
ADAPTER_REPO = "arceus1rk/eligitrace-qwen2.5-1.5b-lora"
ADAPTER_REVISION = "2c3a46e2a52ad90ecec75ffd5aa3380d3c24ceaa"


@dataclass
class Loaded:
    model: torch.nn.Module
    tokenizer: object
    name: str          # human-readable: "base", "ft", "draft"
    model_id: str
    adapter: str | None
    dtype: torch.dtype


def set_threads() -> int:
    n = int(os.environ.get("KORMAT_NUM_THREADS", "0") or 0)
    if n > 0:
        torch.set_num_threads(n)
    return torch.get_num_threads()


def adapter_config():
    """LoraConfig for the shared adapter, readable by our pinned peft (0.13.2).

    The adapter was saved with peft 0.21.2, whose config has extra keys (alora_invocation_tokens,
    lora_bias, use_qalora, ...). Every such key in this adapter is null/false/default, so dropping
    the ones our LoraConfig does not know leaves the adapter's behaviour unchanged; the dropped keys
    are recorded in `.dropped_keys` for provenance.
    """
    import dataclasses
    import json

    from peft import LoraConfig

    raw = json.loads((ADAPTER_DIR / "adapter_config.json").read_text())
    known = {f.name for f in dataclasses.fields(LoraConfig)}
    kept = {k: v for k, v in raw.items() if k in known}
    cfg = LoraConfig(**kept)
    cfg.dropped_keys = {k: raw[k] for k in raw if k not in known}
    return cfg


def _load(model_id: str, dtype: torch.dtype, attn: str):
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, attn_implementation=attn)
    return model, tok


@torch.inference_mode()
def load_target(finetuned: bool = True, merge: bool = True, dtype: torch.dtype = torch.float32,
                attn: str = "sdpa") -> Loaded:
    set_threads()
    model, tok = _load(TARGET_ID, dtype, attn)
    adapter = None
    if finetuned:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(ADAPTER_DIR), config=adapter_config())
        adapter = f"{ADAPTER_REPO}@{ADAPTER_REVISION[:7]}"
        if merge:
            model = model.merge_and_unload()
    model.eval()
    return Loaded(model, tok, "ft" if finetuned else "base", TARGET_ID, adapter, dtype)


@torch.inference_mode()
def load_draft(dtype: torch.dtype = torch.float32, attn: str = "sdpa") -> Loaded:
    set_threads()
    model, tok = _load(DRAFT_ID, dtype, attn)
    model.eval()
    return Loaded(model, tok, "draft", DRAFT_ID, None, dtype)


def eos_ids(tokenizer, model=None) -> set[int]:
    ids: set[int] = set()
    srcs = [tokenizer.eos_token_id]
    if model is not None:
        srcs.append(getattr(model.generation_config, "eos_token_id", None))
    for s in srcs:
        if isinstance(s, int):
            ids.add(s)
        elif s:
            ids.update(int(x) for x in s)
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if isinstance(im_end, int) and im_end >= 0:
        ids.add(im_end)
    return ids


def tiny_qwen(seed: int = 0, vocab: int = 1000):
    """Random 2-layer Qwen2 for fast unit tests (no download)."""
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(seed)
    cfg = Qwen2Config(vocab_size=vocab, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=2048,
                      tie_word_embeddings=False)
    return Qwen2ForCausalLM(cfg).eval()
