"""Repository paths, resolved from this file so scripts work from any cwd."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "results"
DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
CHECKPOINTS = ROOT / "checkpoints"
CONFIGS = ROOT / "configs"
FIGURES = ROOT / "report" / "figures"
TABLES = ROOT / "report" / "tables"
MODEL_LOCK = ROOT / ".model.lock"
