"""Load the dataset, apply the explicit allow-list, cache raw data."""
from __future__ import annotations

import json

from kormat.common.paths import RAW

DATASET = "ncbi/TrialGPT-Criterion-Annotations"
# Explicit allow-list: everything else (gpt4_*, explanation_correctness, training) is dropped.
ALLOW = ["annotation_id", "patient_id", "note", "trial_id", "trial_title", "criterion_type",
         "criterion_text", "expert_sentences", "expert_eligibility"]


def load_raw() -> tuple[list[dict], str]:
    """Returns (allow-listed rows, dataset revision sha). Cached in data/raw."""
    cache, meta = RAW / "annotations.json", RAW / "meta.json"
    if not cache.exists():
        from datasets import load_dataset
        from huggingface_hub import HfApi
        ds = load_dataset(DATASET, split="train")
        rows = [{k: r[k] for k in ALLOW} for r in ds]  # allow-list applied immediately
        RAW.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(rows, indent=0, sort_keys=True))
        meta.write_text(json.dumps({"dataset": DATASET, "revision": HfApi().dataset_info(DATASET).sha,
                                    "n_rows": len(rows), "kept_columns": ALLOW}, indent=2))
    return json.loads(cache.read_text()), json.loads(meta.read_text())["revision"]
