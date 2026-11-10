Partial validation run (fine-tuned model) made BEFORE we noticed that HF `generate` inherits Qwen's
generation_config default `repetition_penalty=1.1` even with `do_sample=False`. Kept as an ablation
(repetition penalty 1.1 vs plain greedy, same rows). The official runs (results/eval/val_ft,
results/eval/test_final/*) use plain greedy: repetition_penalty=1.0, no temperature/top-p/top-k.
