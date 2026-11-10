import json
import os

import numpy as np
import pytest

from kormat.common.paths import PROCESSED, RESULTS
from kormat.data import load_split
from kormat.data.load import ALLOW, load_raw
from kormat.data.split import patient_split

MAN = json.loads((RESULTS / "split_manifest.json").read_text())


def recs(s):
    return [json.loads(l) for l in (PROCESSED / f"{s}.jsonl").read_text().splitlines()]


def test_counts_and_disjoint():
    p = MAN["patients"]
    assert [len(p[s]) for s in ("train", "val", "test")] == [37, 8, 8]
    assert not (set(p["train"]) & set(p["val"]) | set(p["train"]) & set(p["test"]) | set(p["val"]) & set(p["test"]))
    for a in ("train", "val", "test"):
        assert {r["patient_id"] for r in recs(a)} == set(p[a])


def test_each_annotation_once():
    ids = [r["annotation_id"] for s in ("train", "val", "test") for r in recs(s)]
    assert len(ids) == 1015 == len(set(ids))


def test_deterministic_and_matches_manifest():
    rows, _ = load_raw()
    assert patient_split([r["patient_id"] for r in rows]) == MAN["patients"]
    ids = sorted({r["patient_id"] for r in rows})
    np.random.default_rng(42).shuffle(ids)
    assert ids[:37] == MAN["patients"]["train"]


def test_no_gpt4_columns():
    assert not [c for c in ALLOW if c.startswith("gpt4") or c in ("training", "explanation_correctness")]
    for s in ("train", "val", "test"):
        for r in recs(s)[:50]:
            assert not [k for k in r if k.startswith("gpt4") or k in ("training", "explanation_correctness", "note")]


def test_test_seal(monkeypatch):
    monkeypatch.delenv("KORMAT_ALLOW_TEST", raising=False)
    with pytest.raises(PermissionError):
        load_split("test")
    assert len(load_split("train")) == MAN["n_annotations"]["train"]
    assert all(isinstance(k, int) for k in load_split("val")[0]["sentences"])
