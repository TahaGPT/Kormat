"""Load the shared adapter with the pinned peft 0.13.2.

The adapter_config.json was written by peft 0.21.2 and has keys unknown to 0.13.2
(e.g. alora_invocation_tokens), so PeftModel.from_pretrained(dir) raises TypeError. We rebuild the
LoraConfig from the known fields only; the weights are unaffected.
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

from peft import LoraConfig, PeftModel


def load_lora_config(adapter_dir) -> LoraConfig:
    raw = json.loads((Path(adapter_dir) / "adapter_config.json").read_text())
    known = {f.name for f in dataclasses.fields(LoraConfig)}
    dropped = sorted(set(raw) - known)
    cfg = LoraConfig(**{k: v for k, v in raw.items() if k in known})
    cfg._dropped_keys = dropped
    return cfg


def attach_adapter(base_model, adapter_dir):
    cfg = load_lora_config(adapter_dir)
    return PeftModel.from_pretrained(base_model, str(adapter_dir), config=cfg)
