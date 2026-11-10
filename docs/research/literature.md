# Literature review (TGA-4, rubric A1)

17 sources, each opened and verified on 2026-11-10 (arXiv abstract pages, Nature Communications record via PMC/DOI, Crossref for SIGIR 2016). BibTeX keys are in `report/refs.bib`. Numbers quoted are those stated in each paper's abstract or record; we did not re-derive them.

## 1. Clinical eligibility reasoning

| Source | Problem | Method | Key result | Limitation | Take / reject |
|---|---|---|---|---|---|
| Jin et al., Nat. Commun. 15:9074, 2024 (`jin2024trialgpt`) | Matching a patient to trials is slow manual screening. | TrialGPT: retrieval, criterion-level matching with explanations and cited sentences, trial-level ranking, all zero-shot with GPT-4. | Recalls >90% of relevant trials with <6% of the collection; 87.3% criterion-level accuracy on 1,015 expert-annotated patient-criterion pairs (the very pairs in our dataset); 42.6% less screening time in a user study. | Needs a closed, paid, remote LLM; evaluation is on 183 synthetic patients; 1,015 pairs from 53 patients is a small test bed; privacy of real patient notes sent to an API is unresolved. | TAKE the criterion-level task definition, the label set (including "not enough information" and "not applicable") and the sentence-citation output. REJECT the GPT-4 dependency: we adapt a small open model and run locally. |
| Koopman and Zuccon, SIGIR 2016 (`koopman2016test`) | No shared benchmark existed for patient-to-trial matching; prior work used small private data. | Test collection: ~200k ClinicalTrials.gov trials, 60 patient case reports as topics, relevance judged by medical professionals. | Established patient-to-trial matching as a retrieval task with graded judgments. | Judgments are trial-level, topics are short case reports; says nothing about why a single criterion is or is not met. | TAKE the framing and the warning that judgments depend on expert reading. REJECT retrieval as our task; we work at criterion level where TrialGPT annotations exist. |
| Nievas et al., arXiv 2312.09958, 2023 (`nievas2023distilling`) | Proprietary LLMs for trial matching cost money and leak data. | Fine-tune open LLaMA 7B/13B/70B on a GPT-4-generated synthetic dataset. | Reports performance parity with GPT-3.5/GPT-4 (no figures in the abstract). | Training data is synthetic and limited; parity claim is not quantified in the abstract; arXiv preprint, not peer reviewed. | TAKE the premise that fine-tuned open models can close the gap. REJECT synthetic GPT-4 labels: we train on expert labels only and never on the dataset's GPT-4 predictions. |

Critical comparison: TrialGPT shows the task is feasible but with a closed model; Koopman and Zuccon show the retrieval side but not criterion reasoning; Nievas et al. move to open models but with synthetic supervision. None reports evaluation on expert labels with a small (about 1.5B) open model on commodity hardware, and none studies inference cost, which is the gap this project fills.

## 2. Transformer architecture

| Source | Problem | Method | Key result | Limitation | Take / reject |
|---|---|---|---|---|---|
| Vaswani et al., NeurIPS 2017 (`vaswani2017attention`) | Recurrent sequence models are sequential and hard to parallelise. | Scaled dot-product, multi-head self-attention; no recurrence. | 28.4 BLEU on WMT14 En-De; 41.8 BLEU En-Fr single model. | Attention is O(n^2) in time and memory; cache of past K/V grows linearly during decoding. | TAKE the attention formula we reimplement (tiled, E1). |
| Qwen Team, arXiv 2412.15115, 2024 (`qwen2024qwen25`) | Open-weight LLM family across sizes. | Decoder-only Transformer, 18T-token pre-training, post-training; 0.5B to 72B open sizes. | 72B-Instruct competitive with Llama-3-405B-Instruct per the abstract. | Small-model quality in specialised clinical text is not reported there; the report is mostly about large models. | TAKE: Qwen2.5-1.5B-Instruct (target) and 0.5B-Instruct (draft) share one tokenizer (proved in `results/research/tokenizer_compat.json`). |
| Ainslie et al., EMNLP 2023 (`ainslie2023gqa`) | Multi-head attention has a large KV cache; multi-query attention loses quality. | Grouped-query attention: query heads share fewer K/V heads; uptraining from MHA with ~5% of pre-training compute. | Quality close to MHA with speed close to MQA. | Needs (up)training; some quality still lost vs MHA at the same size. | TAKE: Qwen2.5-1.5B uses 12 query / 2 KV heads, so the KV cache is 6x smaller than MHA, which makes our CPU cache practical. |

Critical comparison: Vaswani gives the algorithm; GQA is what makes it deployable on a 15 GB laptop (28,672 bytes/token in bf16 for Qwen2.5-1.5B versus 196,608 for the MHA SmolLM2-1.7B, see `model_comparison.md`); the Qwen report is our evidence that the chosen model is a current, permissively licensed baseline.

## 3. Supervised adaptation

| Source | Problem | Method | Key result | Limitation | Take / reject |
|---|---|---|---|---|---|
| Hu et al., ICLR 2022 (`hu2022lora`) | Full fine-tuning copies all weights per task. | Freeze W; learn W + BA with low rank r; adapters mergeable. | 10,000x fewer trainable parameters and 3x less GPU memory than full fine-tuning of GPT-3 175B, no added inference latency. | Rank and target modules are hyperparameters; may underfit when the domain shift is large; with ~1k training pairs here the evidence of benefit must be measured, not assumed. | TAKE (the shared adapter: r=16, alpha=32 on all linear projections). |
| Dettmers et al., NeurIPS 2023 (`dettmers2023qlora`) | Even LoRA needs the base model in 16-bit memory. | 4-bit NF4 base, double quantisation, paged optimisers, LoRA on top. | 65B model fine-tuned on one 48 GB GPU matching 16-bit performance. | Needs bitsandbytes/CUDA kernels and dequantisation overhead; no CPU path in our pinned stack; unnecessary for a 1.5B model. | REJECT for training (ADR-002). Keep as the reference point in the memory arithmetic. |

Critical comparison: full FT of a 1.5B model needs about 24 GiB for weights, gradients and Adam state in mixed precision (arithmetic in ADR-002), so a parameter-efficient method is mandatory. QLoRA would shave memory further but at a quality/complexity cost we do not need at this size; plain LoRA is sufficient and mergeable into the base.

## 4. Efficient inference

| Source | Problem | Method | Key result | Limitation | Take / reject |
|---|---|---|---|---|---|
| Milakov and Gimelshein, 2018 (`milakov2018online`) | Safe softmax needs separate max and sum passes. | Online normaliser: running max m and sum l updated in one pass. | Up to 1.3x faster softmax; up to 5x for fused Softmax+TopK. | Gains are for GPU memory-bound kernels; arXiv only. | TAKE the (m, l) recurrence: the core of our Algorithm 1. |
| Rabe and Staats, 2021 (`rabe2021selfattention`) | Attention materialises an n x n matrix. | Chunked attention with the same running-sum trick. | O(1) extra memory (O(sqrt n) practical); 59x memory reduction at n=16384 for inference. | No fused kernel, so wall-clock gains are not the point; JAX/TPU. | TAKE the memory argument; our PyTorch CPU tiles follow the same idea. |
| Dao et al., NeurIPS 2022 (`dao2022flashattention`) | Attention is memory-bandwidth-bound. | IO-aware tiling in SRAM with online softmax; exact result. | 15% end-to-end on BERT-large, 3x on GPT-2 (seq 1K). | Gains come from GPU SRAM/HBM hierarchy; a CPU has caches, not a programmable SRAM, so the speedup does not transfer automatically. | TAKE the algorithm; REJECT any claim that it will be faster on our CPU without measuring (E3 reports the honest result). |
| Dao, ICLR 2024 (`dao2024flashattention2`) | FA-1 reaches only 25-40% of peak FLOPs. | Better work partitioning, fewer non-matmul FLOPs. | ~2x over FA-1, 50-73% of A100 peak. | A100/H100-specific parallelism. | TAKE as context only; not reimplemented. |
| Leviathan et al., ICML 2023 (`leviathan2023speculative`) | Decoding is one token per forward pass. | Small draft proposes k tokens; the large model verifies them in one pass; rejection sampling keeps the exact distribution. | 2x-3x on T5-XXL, output identical in distribution. | Needs a good draft; speedup collapses at low acceptance; gains rely on the verify pass costing about the same as one decode step, which is a GPU property. | TAKE: greedy variant (exact-match acceptance) with the Qwen 0.5B draft. |
| Chen et al., 2023 (`chen2023speculative`) | Same, for 70B models. | Speculative sampling with modified rejection scheme; distributed setup. | 2-2.5x on Chinchilla 70B without changing sample quality. | Needs a large family with a small sibling; the evidence is for very large models. | TAKE as a second, independent formulation. |
| Pope et al., MLSys 2023 (`pope2023scaling`) | Inference cost and latency at 500B+ scale. | Partitioning strategies, multi-query attention, int8. | 29 ms/token at low batch (int8), 76% MFU on PaLM 540B; multiquery enables up to 32x longer contexts. | TPU-pod scale, not applicable on a laptop. | TAKE the KV-cache cost model (bytes/token). |
| Kwon et al., SOSP 2023 (`kwon2023pagedattention`) | KV cache fragmentation wastes memory in serving. | PagedAttention: non-contiguous blocks, vLLM. | Near-zero KV waste, 2-4x throughput. | Serving-scale, multi-request benefit; our app serves one request at a time. | REJECT as implementation (we pre-allocate a contiguous cache); cite as the alternative. |

Critical comparison: the three attention papers share one mathematical trick (online softmax), but their benefit is a memory-hierarchy effect on GPUs. On a CPU the fair expectation is lower peak memory and similar or worse speed in eager PyTorch; we state this hypothesis before measuring. Speculative decoding's speedup likewise depends on verification being almost free, which on CPU is only partly true (batch of k+1 tokens is compute-bound earlier), so we measure acceptance rate and wall clock honestly.

## 5. Dataset reference

`trialgpt_criterion_annotations`: the mandated Hugging Face dataset (1,015 criterion annotations, 53 patients). The dataset also holds GPT-4 predictions, which we never use as features.
