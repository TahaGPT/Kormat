# Shared training run: logs of the adapter we use (provenance only)

The LoRA adapter `arceus1rk/eligitrace-qwen2.5-1.5b-lora` (revision 2c3a46e, used for every "fine-tuned" result
in this repository) was trained in a **disclosed joint session on a shared GPU** with M. Ruhan Kamran (23i-0062).
The files here are that run's own logs (`lora_r16_bf16`). They were copied unmodified from the co-trainer's public repository:
https://github.com/ruhank-dev/eligitrace/tree/fe4bf7ed1edb263c0697b6e58420ec8e300d0bdb/results/training/lora_r16_bf16

| file | content |
|---|---|
| `config.yaml` | the run's configuration; identical in every setting to our `configs/train.yaml` |
| `log_history.json` | training loss / lr / grad-norm per logged optimizer step |
| `val_metrics_by_epoch.json` | validation loss and metrics per epoch, **as computed by the shared run's own code** (checkpoint selection: best val macro-F1 → epoch 3) |

These numbers are **not** our results. Every validation and test number in the paper comes from our own evaluation
code (`src/kormat/eval/`) on our own split (identical patient sets; see `../split_overlap_check.json`).
The training-curve figure is plotted by our script `scripts/plot_shared_run.py`.
