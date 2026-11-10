"""Patient-level 37/8/8 split (seed 42)."""
from __future__ import annotations

import numpy as np

SEED = 42
ALGO = ("ids = sorted(df.patient_id.unique()); rng = np.random.default_rng(42); rng.shuffle(ids); "
        "train=ids[:37]; val=ids[37:45]; test=ids[45:]")


def patient_split(patients) -> dict[str, list[str]]:
    ids = sorted(set(patients))
    assert len(ids) == 53, len(ids)
    a = list(ids)
    np.random.default_rng(SEED).shuffle(a)
    b = np.array(ids)
    np.random.default_rng(SEED).shuffle(b)
    assert a == b.tolist(), "list/array shuffle disagree"
    return {"train": a[:37], "val": a[37:45], "test": a[45:]}
