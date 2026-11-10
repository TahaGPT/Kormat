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

## 2026-11-10 · TGA-31 · Attention speed/memory benchmark on CPU
**Goal:** honest runtime and memory comparison (E3). **Machine:** CPU laptop, 4 torch threads, shared with other agents (timings noisy, e.g. tiled_kv_blocked L=256 median 62 ms vs 42 ms at L=512) **Run(s):** results/attention/bench.json, bench.csv; figures report/figures/attn_runtime_vs_L.pdf, attn_memory_vs_L.pdf, attn_runtime_vs_block.pdf
**What we tried:** fp32, d=128, H=12, B=1, non-causal, block 64, warm-up 2, 5 repeats, median. Memory = peak RSS above the pre-call RSS, sampled every 0.5 ms in a fresh subprocess per config (includes output tensor and temporaries); analytical L x L fp32 score matrix = 4*H*L^2 bytes. SDPA backends on CPU: MATH works, FLASH_ATTENTION works (it is the default CPU path), EFFICIENT and CUDNN unavailable.
**Numbers (median s / peak extra MiB; analytical scores MiB in header):**
| L (scores MiB) | ref | tiled_loops | tiled_kv_blocked | torch sdpa | sdpa math |
|---|---|---|---|---|---|
| 1024 (48) | 0.146 / 105 | 0.329 / 7.5 | 0.239 / 43 | 0.148 / 9.8 | 0.193 / 165 |
| 2048 (192) | 0.612 / 401 | 1.076 / 12.3 | 1.081 / 96 | 0.572 / 15.7 | 1.428 / 632 |
| 4096 (768) | 2.957 / 1570 | 5.007 / 25.5 | 3.031 / 175 | 1.942 / 27.7 | 2.674 / 2458 |
Block sweep at L=2048 (loops): 16 -> 7.97 s, 64 -> 1.04 s, 256 -> 0.61 s (12 -> 28 MiB); kv_blocked 16 -> 2.25 s, 128 -> 0.74 s, 256 -> 0.60 s (65 -> 147 MiB).
**Findings:** tiled memory grows roughly linearly in L (loops 7.5 -> 25.5 MiB for L 1024 -> 4096, 3.4x) while ref/math grow ~quadratically (ref 105 -> 1570 MiB, 15x) and exceed the score-matrix bytes by 2-3x because of temporaries (math backend 3.2x at L=4096). Torch's default (flash) CPU kernel matches the tiled memory profile and is fastest at L>=512. Our educational tiled code is NOT faster than ref on CPU: at L=4096 loops is 1.7x slower than ref (kv_blocked about equal), and 2.6x slower than torch SDPA.
**Why educational PyTorch is slower than a fused kernel:** each tile runs ~10 separate eager ops (matmul, mul, mask, amax, maximum, exp, sum, matmul, rescale), each launching with Python/dispatcher overhead, reading and writing its tile to memory (not kept in registers/L1), allocating temporaries, and going through per-op threading. Fused CPU flash kernels keep the tile in cache/registers, fuse exp/max/sum, use vectorised (AVX2) micro-kernels and a single parallel loop over (batch, head, q-block). Small blocks (16) multiply Python iterations (L/16)^2 and made loops 8x slower than block 256. On CPU the L x L matrix mostly fits memory bandwidth, so tiling's benefit here is memory, not speed. CPU has no HBM/SRAM gap as on GPU, which is where FlashAttention gets its speed win.
**What failed / surprised us:** the ref implementation at L=4096 needed 1.6 GB extra; sdpa_math needed 2.5 GB (3.2x scores). Machine contention makes the small-L timings unreliable; use L>=1024 for conclusions.

## 2026-11-10 · TGA-16 · Per-patient prefix KV cache + chunked eval
**Goal:** cut eval cost (824 generations). **Machine:** CPU laptop **Run(s):** results/eval/prefix_cache_check.json
**What we tried:** `--prefix-cache` prefills system+note once per patient (DynamicCache, deep-copied per criterion, passed to generate), token-id prefix verified per record with fallback; `--chunk N` for resumable slices.
**Numbers:** 3 val rows, ft model: outputs identical token-for-token; plain ~10.7 s/example, cached ~7.2 s/example when the prefix is reused (first row of a patient 13.5 s incl. prefill); mean speedup 1.2x on this 3-row sample (prefix 273 tok, suffix 49-70 tok).
**What failed / surprised us:** speedup is capped by decoding (~14 new tokens), not prefill; machine load varies a lot (smoke earlier: 38-86 s/example under contention).

## TGA-2 CPU environment probe (model agent)
- scripts/capture_env.py -> results/env/cpu_laptop.json (i5-10310U, 4C/8T, 15.5 GB, torch 2.5.1 CPU).
- scripts/probe_cpu.py (locked, fresh process per config), results/probe/cpu_probe.csv + summary.md. Merged target fp32: RSS 6.3 GB after load, 9.2 GB peak during load; draft 2.2 GB (3.2 GB peak); target+draft 8.2 GB resident. Prefill@512 about 9 s, decode about 0.5 s/token (target), 0.16 s/token (draft). 4 vs 8 threads differ by 1.1 % (noise) -> KORMAT_NUM_THREADS=4.
- Projection: 412 val+test criteria x about 21 s = about 2.4 h per model pass (assumes 512-token prompt, 24-token answer; timings taken while other agents were running, so indicative). A first probe attempt died at the target+draft step under RAM pressure (nothing recorded); the rerun succeeded.

## TGA-20 Frozen model configuration (model agent)
- configs/final.yaml pins target/draft snapshot shas, adapter repo@revision and the sha256 of adapter_model.safetensors (eb1843b2...), fp32, greedy, max_new_tokens 48.
- Finding: the base generation_config.json ships repetition_penalty=1.1 and sampling defaults; model.generate(do_sample=False) would still apply the penalty. Our checks pass repetition_penalty=1.0; any generate() caller must do the same (hand-written decode loops are unaffected).
- results/finetune/merge_check.json: 5 train prompts, merged vs unmerged greedy outputs identical, max |dlogit| 5.8e-5 (fp32).
- results/finetune/prompt_contract_check.json: 5 train prompts, 5/5 valid {"label","evidence"} JSON with an allowed label (validity only, not accuracy; the 5 shortest of the first 60 train records, label agreement is low in this sample and says nothing about the val metric).

## 2026-11-10 · TGA-24/25/26/27 · No-cache reference, own KV-cache decoder, equivalence grid, KV benchmark
**Goal:** Task 3 (D rubric): a cache-aware decoding flow we wrote ourselves, checked token-for-token against a no-cache reference, and timed. **Machine:** i5-10310U CPU, torch 2.5.1, transformers 4.46.3, shared with other agents (timings noisy). **Runs:** results/kvcache/20261110-193933_24fd5b1_equiv_fp32, results/kvcache/20261110-211114_1fd13f8_bench_kv, results/kvcache/20261110-213732_1fd13f8_bench_nocache; code src/kormat/decoding/{api,cache,workloads}.py, tests/test_kv.py.
**Mechanism (what we read in the installed library and implemented):** the attention layer applies RoPE to q and k from position embeddings and then calls `past_key_value.update(k, v, layer_idx, cache_kwargs)` (modeling_qwen2.py:516-522, cache_kwargs carries cos/sin/cache_position at :522), so keys enter the cache already rotated by their absolute position and are never re-rotated. `PreallocKVCache` (decoding/cache.py) subclasses `transformers.Cache`, owns one pre-allocated (1, kv_heads, max_len, head_dim) buffer per layer and an integer length; update writes a slice and returns a view of the valid prefix. We chose pre-allocation over `torch.cat` (DynamicCache.update, cache_utils.py:441-443, copies the whole cache every token) because a step writes only the new rows and `crop(n)` (cf. DynamicCache.crop :489) is O(1), which speculative rollback needs. A test counts update calls (layers x steps) to prove the model really goes through our class, and the cache asserts that `cache_position` equals its own length. Positions: prefill passes position_ids=arange(P), cache_position=arange(P); a decode step passes position_ids=[[P+t]], cache_position=[P+t], attention_mask=ones(1, P+t+1). Masking: Qwen2Model._update_causal_mask (:933-1008, builder :1010-1072) returns None for a plain prefill or q_len=1 (SDPA then uses is_causal, :1054 region) and otherwise builds a 4-D mask with `arange(target_length) > cache_position[:,None]` masked, where target_length = attention_mask length; that is why the 2-D mask must have length cached+new. A negative-control test feeds a wrong position_id and the logits change by >1e-3, so the equivalence tests are not vacuous.
**Equivalence (fp32, merged fine-tuned target, val-note prompts padded by whole sentences; min prompt is 208 tokens because the frozen template alone is ~190):** 5/5 cells exact token match: P=208 with N=16/64/128, P=523 N=16, P=1025 N=16 (greedy is prefix-closed, so N=16/64 are prefixes of the N=128 run; the grid is capped because the no-cache path is O(n^2) and the machine is shared). Max |dlogit| over the compared steps: 3.9e-5 (N=16), 1.2e-4 (N=64 and 128) at P=208, 3.5e-5 at 523, 4.6e-5 at 1025; step-0 difference is exactly 0.0 (prefill is the same computation in both paths). Differences after step 0 come from different matmul/SDPA shapes (q_len 1 vs full sequence) changing the float summation order, not from a logic difference; no near-tie flipped an argmax, so there is no divergence margin to report. bf16 re-check: see the next entry if present.
**KV benchmark (medians, 3 repeats for kv, 2 for no-cache, 1 warm-up decode, force_len, fp32):**
| P / N | mode | prefill s | decode-only tok/s | total tok/s |
|---|---|---|---|---|
| 208 / 16 | no-cache | 2.75 | 0.31 | 0.31 |
| 208 / 16 | KV | 2.93 | 2.26 | 1.68 |
| 208 / 64 | KV | 2.87 | 2.25 | 2.07 |
| 208 / 128 | KV | 2.88 | 2.23 | 2.14 |
| 523 / 8 | no-cache | 7.39 | 0.13 | 0.13 |
| 523 / 64 | KV | 7.34 | 2.18 | 1.77 |
| 1025 / 4 | no-cache | 13.9 | 0.07 | 0.07 |
| 1025 / 16 | KV | 14.4 | 1.90 | 0.72 |
Decode-only throughput: KV is 7x (P=208), 17x (P=523) and 27x (P=1025) faster than recomputing, and flat in N (2.2-2.3 tok/s at P=208), which matches decoding being weight-bandwidth bound (6 GB of fp32 weights per token) while no-cache pays a full forward over the growing sequence. Prefill is identical for both (the no-cache first step is the prefill). Peak RSS is 6.35 GB at P=208 (weights dominate; sampled every 5 ms); the increment over the call-start RSS is 0 MB at P=208, ~71 MB at P=523, ~105-111 MB at P=1025 (activations plus KV), against an analytical KV size of 12.2 MB (208+16) to 59.6 MB (1025+64) from 2*28 layers*2 kv heads*128*seq*4 B. RSS is process-level, so allocator reuse across cells under-reads some deltas.
**Caveats / surprises:** some cells are noisy because other agents ran models concurrently (P=1025 N=64 prefill 33.5 s vs 14.4 s at N=16; IQR reported in summary.json); no-cache cells were capped (N=4..32) and the time budget skipped larger ones. Figures: report/figures/kv_speed.pdf, kv_memory.pdf.
