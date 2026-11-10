"""Run records: every experiment writes results/<area>/<timestamp>_<gitsha>/ + one line in results/runs.jsonl."""
from __future__ import annotations

import json
import os
import platform
import random
import subprocess
import time
from pathlib import Path

from kormat.common.paths import RESULTS, ROOT


def git_sha(short: bool = True) -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        return out[:7] if short else out
    except Exception:
        return "nogit"


def capture_env() -> dict:
    env = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "hostname": platform.node(),
        "cpu_count": os.cpu_count(),
    }
    try:
        import psutil

        env["ram_total_gb"] = round(psutil.virtual_memory().total / 2**30, 2)
        env["ram_available_gb"] = round(psutil.virtual_memory().available / 2**30, 2)
    except Exception:
        pass
    try:
        import torch

        env["torch"] = torch.__version__
        env["torch_threads"] = torch.get_num_threads()
        env["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            env["gpu"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    for mod in ("transformers", "peft"):
        try:
            env[mod] = __import__(mod).__version__
        except Exception:
            pass
    return env


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(seed)
    except Exception:
        pass


def _append_index(record: dict) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    with (RESULTS / "runs.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def start_run(area: str, config: dict, name: str | None = None) -> Path:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    run_id = f"{stamp}_{git_sha()}" + (f"_{name}" if name else "")
    run_dir = RESULTS / area / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(config, indent=2, default=str))
    (run_dir / "env.json").write_text(json.dumps(capture_env(), indent=2))
    _append_index({"event": "start", "area": area, "run_id": run_id, "time": stamp})
    return run_dir


def finish_run(run_dir: Path, metrics: dict, headline: str | None = None) -> None:
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    _append_index({
        "event": "finish",
        "area": run_dir.parent.name,
        "run_id": run_dir.name,
        "time": time.strftime("%Y%m%d-%H%M%S"),
        "headline": headline,
    })
