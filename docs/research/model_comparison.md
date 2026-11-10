# Model comparison (TGA-5, rubric A2)

Sources: `config.json` fetched from the Hugging Face Hub on 2026-11-10 (`curl -sL https://huggingface.co/<id>/resolve/main/config.json`), parameter totals from the Hub safetensors metadata (`/api/models/<id>`), licences and context from the model cards. **Llama-3.2-1B-Instruct is gated**: its config could not be fetched without accepting Meta's licence, so the figures come from the public byte-identical mirror `unsloth/Llama-3.2-1B-Instruct` (config and safetensors metadata; licence tag `llama3.2`). Treat that column as mirror-derived.

| | Qwen2.5-1.5B-Instruct (chosen) | Qwen3-1.7B | Llama-3.2-1B-Instruct (mirror) | SmolLM2-1.7B-Instruct | Qwen2.5-0.5B-Instruct (draft) |
|---|---|---|---|---|---|
| Parameters (safetensors total) | 1,543,714,304 | 2,031,739,904 (card: "1.7B"; the safetensors total counts embeddings, the card figure follows Qwen's own convention, so we state both) | 1,235,814,400 | 1,711,376,384 | 494,032,768 |
| Layers | 28 | 28 | 16 | 24 | 24 |
| Hidden size | 1536 | 2048 | 2048 | 2048 | 896 |
| Attention heads (Q) | 12 | 16 | 32 | 32 | 14 |
| KV heads | 2 | 8 | 8 | 32 (MHA) | 2 |
| head_dim | 128 (1536/12) | 128 (explicit) | 64 | 64 (2048/32) | 64 (896/14) |
| Vocab | 151,936 | 151,936 | 128,256 | 49,152 | 151,936 |
| Max context | 32,768 | 40,960 | 131,072 | 8,192 | 32,768 |
| RoPE theta | 1,000,000 | 1,000,000 | 500,000 | 130,000 | 1,000,000 |
| tie_word_embeddings | true | true | true | true | true |
| Licence | Apache-2.0 | Apache-2.0 | Llama 3.2 Community Licence (gated, acceptable-use policy) | Apache-2.0 | Apache-2.0 |
| Weights fp32 (GiB) | 5.75 | 7.57 | 4.60 | 6.38 | 1.84 |
| Weights bf16 (GiB) | 2.88 | 3.78 | 2.30 | 3.19 | 0.92 |
| Weights 4-bit (GiB, params x 0.5 B, excl. quant constants) | 0.72 | 0.95 | 0.58 | 0.80 | 0.23 |
| KV bytes / token, bf16 = 2 x L x kv_heads x head_dim x 2 B | 28,672 | 114,688 | 32,768 | 196,608 | 12,288 |
| KV bytes / token, fp32 | 57,344 | 229,376 | 65,536 | 393,216 | 24,576 |
| KV @ 2,048 tokens, bf16 (MiB) | 56.0 | 224.0 | 64.0 | 384.0 | 24.0 |
| Same-tokenizer smaller draft | Qwen2.5-0.5B-Instruct (proved identical) | Qwen3-0.6B (751,632,384 params, Apache-2.0; same family, not verified by us) | none smaller in the family (1B is the smallest) | SmolLM2-360M-Instruct (361,821,120 params; not verified by us) | n/a |

Arithmetic: weights GiB = params x bytes / 2^30 (4, 2, 0.5 bytes). KV bytes/token = 2 (K and V) x layers x kv_heads x head_dim x bytes per element. Example, Qwen2.5-1.5B: 2 x 28 x 2 x 128 x 2 = 28,672. The CPU laptop has 15 GB RAM, no GPU; we run fp32 or bf16 on CPU, and the pre-allocated cache (`src/kormat/decoding/cache.py`) follows the same formula.

## Feasibility on i5-10310U / 15 GB
- Qwen2.5-1.5B in fp32 (5.75 GiB) plus draft 0.5B fp32 (1.84 GiB) = 7.6 GiB, leaving about 7 GiB for activations, cache (56 MiB per 1,024 tokens in fp32: 57,344 x 1,024) and the OS. It fits, but two large models must never load at once beyond this pair (hence the repo's model lock).
- Qwen3-1.7B fp32 (7.57 GiB) fits alone but its KV cache is 4x larger (8 KV heads) and it has a "thinking" mode that would need disabling for a strict-JSON task.
- SmolLM2-1.7B has full multi-head KV (32 heads): 6.9x the cache of Qwen2.5-1.5B and only an 8k context; a patient note plus criterion is a few hundred to ~1,000+ tokens, so 8k is enough, but the cache cost is the worst of the field.

## Tokenizer compatibility proof (Qwen2.5-1.5B vs 0.5B)
`results/research/tokenizer_compat.py` loads the two tokenizers only (no weights) and writes `results/research/tokenizer_compat.json`. Result: vocab dictionaries equal (151,936 entries each, 151,387 merges each), special tokens and added tokens equal, EOS = 151645 and PAD = 151643 in both, chat template and rendered chat identical, `tokenizer.json`, `vocab.json`, `merges.txt` byte-identical (SHA-256 equal), and 0 mismatches on 50 test strings (623 tokens; clinical text, JSON, Unicode, whitespace, empty string). Note: both models have vocab_size 151,936 in config, so the embedding/logit shapes match too, which speculative verification requires. This proves tokenizer compatibility; it does not by itself prove the draft agrees with the target (acceptance rate is measured in F2).
