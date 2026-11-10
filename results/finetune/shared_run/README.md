# Shared training run (adapter arceus1rk/eligitrace-qwen2.5-1.5b-lora)

Status: PARTIAL. The final adapter was produced in a disclosed joint training session on a shared GPU
(with M. Ruhan Kamran, 23i-0062). That run's logs are not in this repository, and no training curve for it
is shown here, because none may be reconstructed or estimated.

What this repository contains instead:
- `configs/train.yaml`: the configuration of the shared run as it was communicated (r=16, alpha=32, dropout 0.05,
  all attention + MLP projections, 3 epochs, lr 2e-4 cosine, 5 % warmup, micro-batch 1 x accum 16, max_len 1024, seed 42).
- `src/kormat/finetune/train.py`: our own implementation of that configuration (CPU smoke run in `../smoke/`).
- `configs/final.yaml`: pinned adapter revision and sha256 of the weights actually used for evaluation.

The user must supply, from the shared run: (1) the training/eval loss log (trainer_state.json or W&B/TensorBoard export),
(2) the checkpoint-selection rule and which step was uploaded, (3) GPU model, wall-clock time, library versions,
(4) confirmation that the exact configuration above is the one that was run, (5) any validation numbers the run reported.
Until then the report must describe the adapter's training only by this configuration and the disclosure.
