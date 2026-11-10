# Kormat demo video: script (target 6 min, allowed 5–7)

Record with OBS (or any screen recorder) at 1080p, then upload to YouTube (unlisted) or Google Drive (anyone with the link). Put the link in the README and the paper (`\todo{video link}` in `report/main.tex`).

**Before recording:** close Brave and other heavy apps. Then run `./scripts/start_app.sh` and wait until the status badge says the models are loaded (1–2 min on CPU), so the recording doesn't show the loading time.

| time | screen | say |
|---|---|---|
| 0:00–0:30 | README on GitHub (logo), then the app | "Kormat screens a patient note against a trial's eligibility criteria, one criterion at a time, and cites the exact patient sentences behind each decision. It is an educational screening prototype, not a clinical decision tool." Point to the disclaimer banner. |
| 0:30–1:00 | terminal: `./scripts/start_app.sh`, then the browser at localhost:8000 | "One command starts it. Qwen2.5-1.5B with our LoRA adapter as the main model and Qwen2.5-0.5B as the draft model, both running on this CPU laptop." Mention that the adapter came from the disclosed joint training session. |
| 1:00–3:00 | Screening tab: paste an **unseen** patient note you wrote yourself, paste a real trial's criteria (or fetch one by NCT ID), click Run | Walk through 3 criteria: the label chip, the cited sentence IDs, clicking a row to highlight the evidence in the note, and the explanation. Show the **missing-information** list and the **overall preliminary outcome** with "how this was decided" (an explicit rule, not ground truth). |
| 3:00–3:45 | invalid inputs: empty note → error; malformed or empty criteria → error; fix them → run works | "Invalid input never crashes the app. It explains what is wrong and keeps working." |
| 3:45–5:15 | Lab tab: choose a criterion, select modes KV-cached / no-cache / speculative (γ=4), run live | Read out prefill/decode ms and tokens/s, and the **token-equality badge** (the cached and speculative outputs equal plain greedy decoding). Give the speculative acceptance rate, and explain why speculative decoding may be slower on a CPU (the draft costs too much relative to the target). Open the stored-results panel. |
| 5:15–6:00 | export: download the HTML report and open it | Show the provenance block (model, adapter hash, git commit, decoding settings) and the disclaimer. Then the History tab: reports survive a restart. |
| 6:00–6:30 | paper PDF, results table | Headline numbers: test macro-F1 base → fine-tuned, evidence F1, KV-cache speed-up, attention max error, speculative acceptance rate. "All code, the split manifest, logs and weights links are in the repository." |

Tips: zoom the browser to 110–125 % so text is readable; keep the mouse still while talking; one take per section is fine (cut them together).
