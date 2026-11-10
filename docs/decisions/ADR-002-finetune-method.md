# ADR-002: Fine-tuning method and attribution

Status: accepted, 2026-11-10.

## Context
Task 1 requires fine-tuning on the 37-patient training split, with the chosen method justified. The laptop has no GPU. A shared GPU was available only in a joint session with M. Ruhan Kamran (23i-0062), where a LoRA adapter was trained and published at `arceus1rk/eligitrace-qwen2.5-1.5b-lora`.

## Options (memory arithmetic, Qwen2.5-1.5B, N = 1,543,714,304 parameters, excluding activations)
| Option | Trainable | Weights | Gradients | Optimiser | Approx. total |
|---|---|---|---|---|---|
| Full fine-tune, AdamW, bf16 weights with fp32 master + fp32 Adam | 1.54B | 2 N = 2.9 GiB | 2 N = 2.9 GiB | fp32 master 4 N + Adam m,v 8 N = 17.3 GiB | 16 N = 23.0 GiB |
| Full fine-tune, pure bf16 AdamW (lossy) | 1.54B | 2.9 GiB | 2.9 GiB | m,v bf16: 5.8 GiB | 11.5 GiB |
| LoRA r=16 on q,k,v,o,gate,up,down | 18,464,768 (1.2%) | base bf16 2.9 GiB | adapter fp32 grads 0.07 GiB | adapter fp32 + Adam 0.14 GiB | about 3.1 GiB |
| QLoRA (NF4 base + LoRA) | 18.5M | base 4-bit about 0.72 GiB + quant constants | 0.07 GiB | 0.14 GiB | about 1.0 GiB |
| Prompt-only (no training) | 0 | 2.9 GiB | 0 | 0 | 2.9 GiB (inference) |

LoRA trainable count: per layer r x (d_in + d_out) summed over q (1536,1536), k (1536,256), v (1536,256), o (1536,1536), gate/up (1536,8960), down (8960,1536): 659,456 per layer x 28 layers = 18,464,768.

Full fine-tuning on 15 GB RAM CPU is impractical both in memory (23 GiB) and time; with only 1,015 annotations (37 training patients) it also risks overfitting and catastrophic drift. QLoRA saves roughly 2 GiB more than LoRA but adds dequantisation cost and bitsandbytes/CUDA dependence for no benefit at 1.5B (Dettmers 2023; reject). Prompt-only is our baseline (B4 "before"), not a final method: the label schema and sentence-ID output must be reliably produced in JSON.

## Decision
LoRA (Hu 2022): r=16, alpha=32, dropout 0.05, targets q,k,v,o,gate,up,down; 3 epochs, lr 2e-4 cosine, effective batch 16, bf16; training loss on the assistant JSON only (see `src/kormat/data/sft.py`, over-long examples are dropped, never silently cut).

## Attribution (disclosed)
- The adapter was trained in a joint session on a shared GPU with M. Ruhan Kamran. It is a shared artefact; we do not claim to have trained it alone, and the report and README say so.
- Our own, independent work: the data pipeline and split, prompt/parse contract, evaluation harness and metrics, the KV-cache decoder, tiled attention, speculative decoder, benchmarks and the web app. Every number in the results was run by us on the CPU laptop.
- The training hyperparameters above are those agreed for the joint run; training curves and logs from that session are to be added from the shared run record when available (open item for the lead). Nothing about the adapter's quality is assumed: before/after metrics are measured by our own evaluation.
- Not copied: no code, results or text from the other student's repository.
