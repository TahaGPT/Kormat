from __future__ import annotations

from pydantic import BaseModel, Field

MAX_NOTE_CHARS = 12000
MAX_CRITERIA = 40
MAX_CRITERION_CHARS = 1000
MODES = ("kv_cached", "no_cache", "speculative", "hf_generate")


class CriterionIn(BaseModel):
    type: str
    text: str


class ScreenRequest(BaseModel):
    note: str = ""
    criteria: list[CriterionIn] = Field(default_factory=list)
    criteria_block: str | None = None
    mode: str = "kv_cached"
    gamma: int = 4
    max_new_tokens: int = 64
    save: bool = True


class LabRequest(BaseModel):
    note: str = ""
    criterion: CriterionIn | None = None
    modes: list[str] = Field(default_factory=lambda: ["no_cache", "kv_cached", "speculative"])
    gamma: int = 4
    max_new_tokens: int = 64


class ParseRequest(BaseModel):
    block: str = ""


class AttentionRequest(BaseModel):
    length: int = 256
