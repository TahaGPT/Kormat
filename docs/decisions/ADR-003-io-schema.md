# ADR-003: Input/output schema and data policy

Status: accepted, 2026-11-10

## Decision
- Input: one prompt per (patient note, criterion). The note is shown as numbered sentences "i. text"; the criterion type, text and the four allowed labels for that type follow (src/kormat/prompting/template.py, frozen).
- Output: JSON only, `{"label": "<allowed label>", "evidence": [<sentence ids>]}`. Labels keep the expert's four categories per criterion type (never collapsed to yes/no). Invalid output is scored as INVALID.
- Prompt contract is byte-identical to the one the shared LoRA adapter (arceus1rk/eligitrace-qwen2.5-1.5b-lora, disclosed joint training session) was trained with; tests/test_template.py pins it (golden user message, system-prompt hash, target format).
- SFT: loss on the completion only (prompt labels = -100); target = JSON + `<|im_end|>`; the prompt is asserted to be an exact token prefix of the full chat.
- Data: allow-list of columns; all gpt4_* fields, explanation_correctness and `training` are dropped at load. Split by patient (37/8/8, seed 42), recorded in results/split_manifest.json. Test split is sealed behind KORMAT_ALLOW_TEST=1 and every access is logged.
- Length: no example exceeds 622 tokens (max_len 1024), so no truncation occurs; policy if it did: drop, never cut (see results/data/token_stats.json).
- Known data defect: annotation 883 (train) has a null criterion_text; it is kept in the split counts but skipped by SFT.

## Why
Patient-level splitting prevents leakage across criteria of one note; a single frozen prompt keeps training, evaluation, benchmarks and the app comparable.
