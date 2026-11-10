# ADR-004: Application architecture and aggregation rule

Status: accepted (2026-11-10)

## Context
Task 6 needs one browser-accessible app: supply a note, enter or select criteria, run the model, inspect per-criterion labels and evidence, compare decoding mechanisms, and export a report. It must run on a CPU laptop and be demonstrable without code.

## Decisions
- **Backend**: FastAPI (`kormat.app.server`), one process, one generation at a time (engine lock). Models (target 1.5B + merged LoRA, draft 0.5B) load in a background thread at startup; `/api/health` exposes `idle|loading|ready|error` and progress; screening during loading returns 503 with a retry hint.
- **Frontend**: static HTML/CSS/JS, no build step, Chart.js 4.4.1 vendored locally so it works offline.
- **Fake engine** (`KORMAT_FAKE_ENGINE=1`): deterministic, model-free; used by tests and layout demos only, never for reported numbers.
- **Persistence**: SQLite `app_data/kormat.db` stores each report (note, criteria, labels, evidence, outcome, provenance) so history survives restarts.
- **Provenance** on every report: model id, adapter id@revision, dtype, git sha, decoding mode and settings, timestamp.
- **Validation**: friendly JSON errors `{error:{code,message,hint}}` for empty note, note > 12000 chars, no criteria, > 40 criteria, bad type, unknown mode, bad gamma, model loading (503), and uncaught exceptions (500).
- **Test split stays sealed**: samples come from train/val only.

## Aggregation rule (preliminary outcome)
1. Any exclusion criterion labelled `excluded` -> **Likely ineligible**.
2. Else any inclusion criterion labelled `not included` -> **Likely ineligible**.
3. Else any `not enough information` or INVALID (unparseable) -> **Undetermined, information missing**.
4. Else -> **Potentially eligible**.
`not applicable` is ignored in the decision but listed.

Rationale: a single confirmed disqualifier is decisive; absence of information must never be read as eligibility, so unknowns block "Potentially eligible". The rule is deterministic and shown in the UI, the report and the paper. It is a screening aid computed from model output, never ground truth and never an enrollment decision; the model's labels can be wrong.

## Consequences
The rule is conservative (many "Undetermined" outcomes on short notes), which is the intended failure direction. Per-criterion accuracy is evaluated separately (not by this rule).
