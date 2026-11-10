# Kormat: research journey

This is the single progress log for GenAI Assignment 2. It records where we started, where we are now, what we tried, what failed, the numbers we got, and the decisions we made. The IEEE paper is written from it.
Board: https://synverk.abdulhaque.dev/projects/aee8a2fc-a70e-4570-8c66-cfdc59bc0519 · Repo: https://github.com/TahaGPT/Kormat

## Status board (rubric → evidence)
| code | criterion | cards | status | evidence |
|---|---|---|---|---|
| A1 | literature | TGA-4 | todo | docs/research/literature.md |
| A2 | model choice & feasibility | TGA-2, TGA-5 | todo | docs/research/model_comparison.md |
| A3 | research-informed decisions | TGA-5, TGA-6, ADRs | todo | docs/decisions/ |
| B1 | dataset integrity & fixed split | TGA-7, TGA-9, TGA-3 | todo | results/split_manifest.json |
| B2 | I/O processing | TGA-8, TGA-10, TGA-11, TGA-12 | todo | results/data/ |
| B3 | fine-tuning implementation | TGA-18, TGA-19, TGA-20 | todo | results/finetune/ |
| B4 | before/after | TGA-17, TGA-21 | todo | results/eval/ |
| C1–C3 | labels, detailed metrics, evidence | TGA-13–16, TGA-22 | todo | results/eval/test_final/ |
| C4 | traceable explanations, ≥8 errors | TGA-23 | todo | docs/research/error_analysis.md |
| D1–D3 | KV cache | TGA-24–27 | todo | results/kvcache/ |
| E1–E3 | attention | TGA-28–31 | todo | results/attention/ |
| F1–F3 | speculative | TGA-32–35 | todo | results/speculative/ |
| G1–G4 | application | TGA-36–41 | todo | src/kormat/app/, results/app/ |
| H1–H3 | paper, repro, defense | TGA-42–48 | todo | report/ |

## Entry format
```
## YYYY-MM-DD · TGA-n · <title>
**Goal:** … **Machine:** CPU laptop **Run(s):** results/…/
**What we tried:** …
**Numbers:** (table)
**What failed / surprised us:** …
**Decision:** … (ADR-00x)
**Paper hook:** section / figure this feeds
```

---

## 2026-11-08 · Planning · assignment analysis and board
**Goal:** break the brief into rubric-mapped cards (48 cards on Synverk, TGA-1…48).
**Decisions:** Qwen2.5-1.5B-Instruct as the main model and Qwen2.5-0.5B-Instruct as the draft (same tokenizer, Apache-2.0, GQA → small KV cache); parameter-efficient fine-tuning; FastAPI + static frontend.
**Paper hook:** §I contributions, §III overview.

## 2026-11-10 · Plan change · CPU-only execution + shared adapter
**What changed:** the planned 4 GB GPU (GTX 1650 / RTX 3050) laptop route was dropped. Fine-tuning was done in a joint session on a shared GPU together with M. Ruhan Kamran (23i-0062). The resulting LoRA adapter (`arceus1rk/eligitrace-qwen2.5-1.5b-lora`) is used and **disclosed**. Everything else (data, evaluation, KV cache, attention, speculative decoding, app, every reported number) is implemented and run independently on the CPU laptop (i5-10310U, 4C/8T, 15 GB RAM).
**Consequence:** the adapter's training prompt becomes our fixed prompt contract (`src/kormat/prompting/template.py`). A split-overlap check (TGA-3) is mandatory before any evaluation.

## 2026-11-10 · TGA-1 · repo scaffold
**What we did:** uv project (Python 3.10, CPU torch 2.5.1, transformers 4.46.3, peft 0.13.2, all pinned), feature folders under `src/kormat/`, run-record convention (`kormat.common.runlog`: every run → `results/<area>/<stamp>_<sha>/{config,env,metrics}.json` + `results/runs.jsonl`), Makefile with a model lock (`flock .model.lock`) so only one large model loads at a time on 15 GB RAM.

## 2026-11-10 · TGA-4/5/6 · literature, model comparison, ADR-001/002
**Goal:** rubric A1-A3 evidence. **Machine:** CPU laptop (tokenizer-only work, no weights loaded). **Run(s):** results/research/tokenizer_compat.json
**What we did:** verified 17 sources against arXiv/Nature/Crossref records (docs/research/literature.md, report/refs.bib); pulled config.json for Qwen2.5-1.5B/0.5B, Qwen3-1.7B, SmolLM2-1.7B and (gated Llama-3.2-1B via the unsloth mirror) and computed weight/KV memory (docs/research/model_comparison.md).
**Numbers:** Qwen2.5-1.5B 1,543,714,304 params, 28 layers, GQA 12/2, KV 28,672 B/token bf16 (SmolLM2 196,608; Qwen3-1.7B 114,688). Tokenizers of 1.5B and 0.5B: identical vocab, merges, special tokens, chat template, file hashes; 0 id mismatches on 50 strings.
**What surprised us:** Qwen3-1.7B card says 1.7B but safetensors total is 2.03B (embeddings counted). Llama-3.2 is gated. LoRA trainable params = 18,464,768 (1.2%) by arithmetic.
**Decision:** Qwen2.5-1.5B + 0.5B draft (ADR-001); LoRA, disclosed joint training (ADR-002). Caveat recorded: base model was fixed by the shared run; other candidates not evaluated by us.
**Paper hook:** §II Related Work, §III Materials & Methods (model selection, fine-tuning).

## 2026-11-10 · TGA-7/8/9/10/11/12/3 · Data, split, token stats, SFT builder
**Goal:** leakage-free fixed split and I/O pipeline. **Machine:** CPU laptop **Run(s):** results/data/, results/split_manifest.json, results/finetune/
**What we tried:** allow-listed load of ncbi/TrialGPT-Criterion-Annotations (rev 1cfcafde), patient split seed 42, numbered-note parser + pysbd splitter, SFT builder with completion-only mask, golden prompt test.
**Numbers:** 1015 annotations; train/val/test = 603/206/206 annotations over 37/8/8 patients; tokens per example median 303/344/303, max 622/456/516, none over 1024. Gold validation: 0 bad evidence ids, 0 bad labels. Split patient sets identical to the joint-session split (results/finetune/split_overlap_check.json). Adapter: 18,464,768 params, r=16, alpha=32, all 7 projections.
**What failed / surprised us:** annotation 883 has a null criterion_text (train); skipped by SFT. Label skew: exclusion is 484/659 "not excluded"; inclusion "not included" only 23.
