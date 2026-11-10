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

## 2026-11-10 · TGA-13/14/15/16 · Parser, metrics, evidence metrics, eval runner
**Goal:** robust output parsing and reproducible evaluation. **Machine:** CPU laptop **Run(s):** results/eval/smoke_*/
**What we tried:** parse.py (think/fence/single-quote/synonym handling, type-mismatch -> INVALID, hallucinated ids split out); classification.py (fixed label lists, INVALID predicted-only column, bootstrap CI, confusion heatmaps); evidence.py (explicit empty-set rules, four-cell counts); explain.py (explanations only from cited sentences); run.py (resumable per-row jsonl, sealed test split); errors.py (auto-tagging for TGA-23).
**Numbers:** 17 unit tests incl. hand-computed toy metrics and sklearn cross-check pass; real-model numbers pending smoke run.
**What failed / surprised us:** loading the shared adapter with peft 0.13.2 raised TypeError on unknown LoraConfig key `alora_invocation_tokens` (adapter saved by a newer peft).

## 2026-11-10 · TGA-43/44 · IEEE LaTeX skeleton and methods sections
**Goal:** rubric H1 scaffold. **What we did:** report/main.tex (IEEEtran conference), sections/*.tex, five TikZ diagrams (pipeline, KV flow, tile loop, speculative loop, app flow), Algorithm 1 (tiled online softmax) and Algorithm 2 (greedy speculative), AI-use appendix with joint-training disclosure, Introduction / Related Work / Methods / Experimental Setup written from the code. Generated tables load via \maybeinput so missing ones show a red TODO instead of breaking the build.
**What failed / surprised us:** no TeX distribution on the laptop (pdflatex/latexmk not found), so the paper is NOT compiled yet; brace/env balance and cite-key coverage were checked by script. Use Overleaf (upload report/) or install TeX Live, then `cd report && make`.
**Open:** Results section is a skeleton; methods text must be re-checked against the final code (spec decoder, app, eval evidence rules) and the shared run's training logs.
**Paper hook:** whole report.

## 2026-11-10 · TGA-36..41 · application (engine, API, UI, lab, export, history, startup)
**Goal:** one browser app for Task 6. **Machine:** CPU laptop **Run(s):** results/app/
**What we built:** `kormat.app` (engine with fake/real modes and a generation lock, FastAPI server with friendly validation errors, aggregation rule, criteria parser, SQLite history, print-friendly HTML report + JSON), static UI (dark/light, disclaimer banner, evidence highlighting, lab with live charts and token-equality badge, stored-results panel), `scripts/start_app.sh`. Tests run on the fake engine: aggregation, parser, every validation error, recovery after exceptions, offline NCT fetch, restart persistence.
**Decision:** conservative aggregation rule, unknowns never count as eligibility (ADR-004).
**What surprised us:** RAM is shared with other agents, so real-model smoke runs queue behind the model lock; UI layout was verified with the fake engine (screenshots in report/figures/app_*.png are fake-engine layouts, not results).
**Paper hook:** §III application workflow; screenshots figure.

## 2026-11-10 · TGA-28/29/30 · Reference SDPA, tiled online-softmax attention, numerical validation
**Goal:** correct, validated attention kernels (E1, E2). **Machine:** CPU laptop **Run(s):** results/attention/validation.csv, summary.json
**What we tried:** `attention_ref` (full L x L scores, max-subtracted softmax, GQA via repeat_interleave, bottom-right causal mask `j <= i + Lk - Lq`, bool mask, fully masked rows give 0). `tiled_attention` in two variants: "loops" (explicit q-block x kv-block Python loops) and "kv_blocked" (all queries, loop over kv blocks); fp32 accumulation; causal block skipping; guard so `-inf - -inf` never yields NaN.
**Online-softmax derivation (paper Algorithm 1):** for a row with scores s_1..s_n, softmax output is o = (sum_j e^{s_j - M} v_j) / (sum_j e^{s_j - M}) for any constant M. Process keys in blocks. After blocks 1..t keep m_t = max so far, l_t = sum_{j<=t} e^{s_j - m_t}, O_t = sum_{j<=t} e^{s_j - m_t} v_j. A new block with block max m~ gives m_{t+1} = max(m_t, m~). Then e^{s_j - m_{t+1}} = e^{m_t - m_{t+1}} e^{s_j - m_t} for old j, so l_{t+1} = e^{m_t - m_{t+1}} l_t + sum_{new} e^{s_j - m_{t+1}} and O_{t+1} = e^{m_t - m_{t+1}} O_t + sum_{new} e^{s_j - m_{t+1}} v_j. The old terms are rescaled exactly by one scalar per row, so after the last block o = O_n / l_n equals full softmax attention and no L x L matrix is stored. First block: m_0=-inf, e^{-inf}=0 (we use m_safe=0 when m_new=-inf to avoid NaN). Logsumexp = m + log l.
**Numbers:** 1680 rows (3 impls x 560 cases: L in {1,7,16,63,64,65,127,256,513,1024}, Lq!=Lk (1,777),(5,300), d 64/128, B 1/2, (H,Hkv) (4,4)/(12,2), blocks 16-128 incl. asymmetric, 4 mask kinds, fp32/bf16/fp16), all pass vs float64 truth.
| dtype | max_abs | tol (atol, rtol) |
|---|---|---|
| fp32 | 2.5e-6 | 1e-5, 1e-4 |
| fp16 | 9.7e-4 | 2e-3, 2e-3 |
| bf16 | 7.8e-3 | 2e-3, 8e-3 (bf16 output rounding is one ulp = 2^-7 relative) |
Stability: at logits x1000 (max |logit| 4543) naive exp/sum gives NaN, tiled output is finite with max abs error 3e-4 vs float64.
**What failed / surprised us:** with the brief's bf16 rtol 2e-3 the output rounding alone would exceed tolerance, so rtol was set to 8e-3 and the reason recorded. max_rel column is large (up to 0.12) only where the true value is ~0 (floored at 1e-6); the pass criterion is the atol+rtol rule. fp16 works on CPU (inputs rounded to half, compute in fp32). Truth uses the already-rounded half inputs so only kernel error is measured.
