<p align="center"><img src="assets/logo.png" alt="Kormat" width="420"></p>

# Kormat: evidence-traced clinical-trial eligibility screening

> **Educational screening prototype, not a clinical enrollment decision tool.** Kormat must never be used to make medical or enrollment decisions.

Kormat takes **a patient note**, **a criterion type** (inclusion/exclusion) and **the criterion text**. For each criterion it returns:
- a structured eligibility label (`included`, `not included`, `excluded`, `not excluded`, `not enough information`, `not applicable`)
- the IDs of the patient sentences that support the label, with the evidence text
- an explanation built only from the cited sentences.

It is built on `Qwen/Qwen2.5-1.5B-Instruct` with a LoRA adapter, and has three hand-written inference mechanisms:
- a KV-cached decoder, checked against a no-cache reference
- tiled online-softmax attention
- greedy speculative decoding with `Qwen/Qwen2.5-0.5B-Instruct` as the draft model.

A browser app puts it all together. It runs entirely on a **CPU laptop**.

## Status
Submission for GenAI Assignment 2 (10 Nov 2026). The full research log, from the first plan to the final numbers, is [`docs/JOURNEY.md`](docs/JOURNEY.md). Design decisions are recorded in [`docs/decisions/`](docs/decisions/) and the paper is [`report/main.pdf`](report/main.pdf).

## Hardware used
Everything except the LoRA training ran on a **CPU laptop**: Intel i5-10310U (4 cores / 8 threads), 15 GB RAM, no GPU, Ubuntu 22.04, Python 3.10.
- Memory: the merged fine-tuned 1.5B model in fp32 needs about 6.3 GB RSS; target and draft together about 8.2 GB (`results/probe/summary.md`).
- Speed: decoding runs at about 2.2 tokens/s with the KV cache.

## Setup
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # if uv is missing
git clone https://github.com/TahaGPT/Kormat && cd Kormat
uv sync                                           # Python 3.10 + CPU torch 2.5.1, transformers 4.46.3, peft 0.13.2 (pinned in uv.lock)
make download-weights                             # Qwen2.5-1.5B/0.5B-Instruct + LoRA adapter (pinned revisions)
make test                                         # fast unit tests (no model download needed)
```

## Run the app (no code editing needed)
```bash
./scripts/start_app.sh          # then open http://localhost:8000
```
- Models load in the background, which takes about 1–2 minutes on CPU; the status badge shows progress.
- Close heavy apps first: the app needs about 8 GB RAM.
- `KORMAT_FAKE_ENGINE=1 ./scripts/start_app.sh` starts a model-free demo with deterministic fake results. Use it for the UI only.

What you can do in the app:
- **Screening:** paste a patient note (any free text; it is split into numbered sentences), paste a trial's eligibility block or fetch it by NCT ID, choose a decoding mode, and run. Each criterion shows its label, the cited sentence IDs (click to highlight them in the note) and an evidence-only explanation. The page also lists missing information and gives a rule-based preliminary outcome.
- **Lab:** compares no-cache, KV-cached, speculative (γ) and HF decoding live on the same input: latency, tokens/s, memory, acceptance rate, and a token-equality check.
- **Export / History:** print-friendly HTML and JSON reports with provenance (model, adapter hash, git commit, decoding settings); history is kept in SQLite.

## Reproduce the experiments
| step | command | output |
|---|---|---|
| data + split + manifest | `make data` | `data/processed/`, `results/split_manifest.json`, `results/data/` |
| evaluation (base vs fine-tuned, val + test) | `make eval-all` (chunked, resumable; takes hours on CPU) | `results/eval/{test_final,val_ft,val_base}/` |
| KV cache: equivalence + benchmark | `uv run python scripts/kv_equivalence.py`, `make bench-kv` | `results/kvcache/` |
| attention: validation + benchmark | `uv run python scripts/attn_validate.py`, `make bench-attn` | `results/attention/` |
| speculative: equivalence + benchmark | `uv run python scripts/spec_equivalence.py`, `make bench-spec` | `results/speculative/` |
| training script (CPU smoke run on 0.5B) | `make train-smoke` | `results/finetune/smoke/` |
| tables + figures | `make figures` | `report/tables/`, `report/figures/` |
| paper | `make paper` (latexmk, or Tectonic if LaTeX is not installed) | `report/main.pdf` |

Every run writes `config.json` + `env.json` + `metrics.json` under `results/<area>/<timestamp>_<gitsha>/` and a line in `results/runs.jsonl`. Every access to the sealed test split is logged in `results/test_access.log`.

## Decoding settings
All reported numbers use plain greedy decoding: `do_sample=False`, `repetition_penalty=1.0`, `max_new_tokens=48`, fp32, batch 1. Qwen's own `generation_config.json` ships `repetition_penalty=1.1`, which HF applies even in greedy mode, so we override it explicitly (see JOURNEY).

## Weights and disclosure
The LoRA adapter was trained in a joint session on a shared GPU together with M. Ruhan Kamran (23i-0062) and is published at <https://huggingface.co/arceus1rk/eligitrace-qwen2.5-1.5b-lora>. All data processing, evaluation, decoding, attention, application code, and every reported result in this repository were implemented and produced independently.

## Data
[`ncbi/TrialGPT-Criterion-Annotations`](https://huggingface.co/datasets/ncbi/TrialGPT-Criterion-Annotations) (Jin et al., TrialGPT). GPT-4 prediction/explanation columns are never used as inputs or in evaluation.
