"""Sentence-ID handling: parse numbered notes, split free text (pysbd), unify."""
from __future__ import annotations

import re

_NUM = re.compile(r"^\s*(\d+)\s*\.\s*(.*)$")
_BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")


def is_numbered(text: str) -> bool:
    lines = [l for l in text.splitlines() if l.strip()]
    return bool(lines) and _NUM.match(lines[0]) is not None and int(_NUM.match(lines[0]).group(1)) == 0


def parse_numbered_note(note: str) -> dict[int, str]:
    out: dict[int, str] = {}
    cur = None
    for line in note.splitlines():
        m = _NUM.match(line)
        if m and (cur is None and int(m.group(1)) == 0 or cur is not None and int(m.group(1)) == cur + 1):
            cur = int(m.group(1))
            out[cur] = m.group(2).strip()
        elif cur is not None and line.strip():
            out[cur] = (out[cur] + " " + line.strip()).strip()
    return out


_SEG = None


def split_free_text(text: str) -> dict[int, str]:
    global _SEG
    if _SEG is None:
        import pysbd
        _SEG = pysbd.Segmenter(language="en", clean=False)
    sents: list[str] = []
    for block in re.split(r"\n\s*\n|\n(?=\s*(?:[-*•]|\d+[.)])\s)", text.strip()):
        block = _BULLET.sub("", " ".join(block.split()))
        if block:
            sents.extend(s.strip() for s in _SEG.segment(block) if s.strip())
    return dict(enumerate(sents))


def ensure_sentences(text: str) -> dict[int, str]:
    return parse_numbered_note(text) if is_numbered(text) else split_free_text(text)
