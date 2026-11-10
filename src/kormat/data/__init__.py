"""Data access. `load_split` is the only sanctioned way to read processed splits."""
from __future__ import annotations

import json
import os
import time

from kormat.common.paths import PROCESSED, RESULTS

SPLITS = ("train", "val", "test")


def load_split(name: str) -> list[dict]:
    if name not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {name!r}")
    if name == "test":
        if os.environ.get("KORMAT_ALLOW_TEST") != "1":
            raise PermissionError("test split is sealed; set KORMAT_ALLOW_TEST=1 for the final evaluation only")
        RESULTS.mkdir(parents=True, exist_ok=True)
        with open(RESULTS / "test_access.log", "a") as f:
            f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')}\tpid={os.getpid()}\targv={' '.join(os.sys.argv)[:200]}\n")
    recs = []
    for line in (PROCESSED / f"{name}.jsonl").read_text().splitlines():
        r = json.loads(line)
        r["sentences"] = {int(k): v for k, v in r["sentences"].items()}
        recs.append(r)
    return recs
