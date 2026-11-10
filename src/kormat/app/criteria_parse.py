"""Parse a pasted eligibility-criteria block into [{type, text}]."""
from __future__ import annotations

import re

_HDR = re.compile(r"^\s*(?:[-*#\s]*)(inclusion|exclusion)\s+criteria\s*:?\s*[-*#]*\s*$", re.I)
_HDR_INLINE = re.compile(r"^\s*(inclusion|exclusion)\s+criteria\s*:\s*(.+)$", re.I)
_BULLET = re.compile(r"^\s*(?:[-*•●·]+|\(?\d{1,3}[.)]|\(?[a-zA-Z][.)])\s+")


def parse_criteria(block: str, default_type: str = "inclusion") -> list[dict]:
    out: list[dict] = []
    lines_all = (block or "").replace("\r", "").split("\n")
    has_bullets = any(_BULLET.match(l) for l in lines_all)
    cur = default_type
    buf: list[str] | None = None

    def flush():
        nonlocal buf
        if buf:
            t = " ".join(s.strip() for s in buf).strip()
            if t:
                out.append({"type": cur, "text": t})
        buf = None

    for raw in lines_all:
        line = raw.rstrip()
        if not line.strip():
            flush()
            continue
        m = _HDR.match(line)
        mi = _HDR_INLINE.match(line)
        if m or mi:
            flush()
            cur = (m or mi).group(1).lower()
            if mi and not m:
                buf = [mi.group(2)]
            continue
        if _BULLET.match(line):
            flush()
            buf = [_BULLET.sub("", line, count=1)]
        elif not has_bullets:
            flush()
            buf = [line]
        elif buf is not None and raw[:1] in (" ", "\t"):
            buf.append(line)  # indented continuation
        elif buf is not None:
            buf.append(line)  # wrapped line
        else:
            buf = [line]
    flush()
    return out
