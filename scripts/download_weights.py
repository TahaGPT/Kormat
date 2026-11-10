"""Fetch base, draft and the shared LoRA adapter (pinned revisions) into the HF cache / checkpoints/."""
from huggingface_hub import snapshot_download

from kormat.common.paths import CHECKPOINTS

BASE = ("Qwen/Qwen2.5-1.5B-Instruct", None)
DRAFT = ("Qwen/Qwen2.5-0.5B-Instruct", None)
ADAPTER = ("arceus1rk/eligitrace-qwen2.5-1.5b-lora", "2c3a46e2a52ad90ecec75ffd5aa3380d3c24ceaa")
PATTERNS = ["*.json", "*.safetensors", "*.txt", "merges.txt", "*.jinja"]


def main() -> None:
    for repo, rev in (BASE, DRAFT):
        p = snapshot_download(repo, revision=rev, allow_patterns=PATTERNS)
        print(f"{repo} -> {p}")
    p = snapshot_download(ADAPTER[0], revision=ADAPTER[1], local_dir=CHECKPOINTS / "adapter")
    print(f"{ADAPTER[0]}@{ADAPTER[1][:7]} -> {p}")


if __name__ == "__main__":
    main()
