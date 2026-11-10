# ADR-001: Target and draft model choice

Status: accepted, 2026-11-10. Evidence: `docs/research/model_comparison.md`, `results/research/tokenizer_compat.json`, `docs/research/literature.md`.

## Context
Need an open-weight decoder that (a) is small enough to run, evaluate and decode with our own KV cache on a CPU laptop with 15 GB RAM and no GPU, (b) has a smaller sibling sharing its tokenizer for speculative decoding, (c) is permissively licensed, (d) has a context long enough for a patient note plus criterion, and (e) is one on which the shared LoRA adapter exists.

## Options considered
1. Qwen2.5-1.5B-Instruct (+0.5B draft). 1.54B params, 28 layers, GQA 12/2, 32k context, Apache-2.0, 28,672 B/token KV (bf16).
2. Qwen3-1.7B. 2.03B counted params, GQA 16/8 (4x the KV of option 1), 40k context, Apache-2.0, thinking mode to suppress.
3. Llama-3.2-1B-Instruct. 1.24B params, 128k context, smaller KV than Qwen3 but 16 layers of width 2048; gated community licence; no smaller same-tokenizer model, so no draft.
4. SmolLM2-1.7B-Instruct. 1.71B, multi-head KV (196,608 B/token), 8k context, Apache-2.0; draft available (360M).

## Decision
Qwen2.5-1.5B-Instruct as target, Qwen2.5-0.5B-Instruct as draft.

Reasons, each tied to evidence: smallest KV cache per parameter among the 1.5-2B options (6.9x smaller than SmolLM2, 4x smaller than Qwen3) which directly serves the cache and CPU-memory goals (GQA, Ainslie 2023); verified identical tokenizer with an in-family 0.5B draft (0 mismatches on 50 strings, identical vocab/merges/template files); Apache-2.0 with no gating (Llama's gate would block a reviewer reproducing the setup); 5.75 GiB fp32 weights fit on the laptop together with the 1.84 GiB draft; and the LoRA adapter we are allowed to use was trained on this base in the joint session (ADR-002), so the choice is also constrained by what exists.

## Consequences and honest caveats
- The choice is not purely independent: the base was fixed by the shared training run. The comparison above was done to document that the choice is defensible, and it was in fact the planned choice since 2026-11-08 (see JOURNEY planning entry), but the other candidates were not fine-tuned or evaluated by us. We make no claim that Qwen2.5-1.5B beats them on this task.
- We did not measure Qwen3, Llama or SmolLM2 on the task; arguments about them rest on config arithmetic and model cards.
- Llama figures come from a public mirror because the original is gated.
- Parameter count 1,543,714,304 includes the tied embedding once (151,936 x 1,536 = 233M, 15% of the model).
