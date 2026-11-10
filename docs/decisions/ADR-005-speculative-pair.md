# ADR-005: Draft/target pair for speculative decoding
- **Date:** 2026-11-10  **Card:** TGA-32  **Status:** accepted
## Context
Task 5 needs a smaller draft model with a tokenizer compatible with the target (Qwen2.5-1.5B-Instruct + shared LoRA, merged, fp32). Greedy speculative decoding is only exact if draft and target share token ids.
## Options considered
| option | evidence | verdict |
|---|---|---|
| Qwen2.5-0.5B-Instruct (plain, not fine-tuned) | results/speculative/pair_tokenizer.json (tests/test_spec_pair.py): vocab dicts identical (151665 tokens), config vocab_size 151936 for both, added tokens, EOS ids ({151645}) and chat template identical, probe encodings identical for target, draft and adapter tokenizer; 24 layers vs 28, hidden 896 vs 1536, both 2 KV heads | chosen |
| Fine-tune a draft on the same adapter recipe | not done (time) | rejected for the deadline; would likely raise acceptance |
| Different family (e.g. SmolLM2) | different tokenizer, ids do not map | rejected |
## Decision
Draft = Qwen2.5-0.5B-Instruct fp32. Memory budget (measured, results/probe and results/speculative/*bench_A): target ~6.3 GB RSS, draft ~2.2 GB, both resident ~8.3 GB peak process RSS; KV for both caches ~27 MB at P~330. gamma grid {1,2,3,4,6}.
Measured c (draft/target single-token decode-step time at cache length 330, fp32, loaded machine): **c = 0.337** (146 ms vs 434 ms), results/speculative/20261110-215517_8fbabb1_bench_A/summary_A.json.
## Consequences
On this CPU a verify forward of q_len>=2 costs far more than a q_len=1 step: measured 731 ms (q=2) .. 764 ms (q=9) vs 434 ms for q=1, i.e. v~1.7 decode steps, instead of the v=1 assumed by the (1-a^(g+1))/((1-a)(gc+1)) formula. Micro-benchmark (results/speculative/micro_linear.json): the lm_head (151936x1536) takes 74 ms at M=1 (gemv) but 209-276 ms for M=2..9 (gemm), and the 8960x1536 MLP matrix 4.9 vs 12-14 ms. So the formula is plotted both with c only and with the measured v; only part of the q=1 -> q=2 jump (~300 ms) is explained by this lm_head effect, the rest was not isolated. The real task output is only ~14 tokens, so prefill (~7 s of target + ~2 s draft) dominates and end-to-end speedup is ~1.0x.
## Rejected alternatives and why
Sampling-based (rejection-sampling) speculation: the assignment asks for greedy; exactness is checked by token equality.
