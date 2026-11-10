# Error analysis: fine-tuned model on the sealed test split (TGA-23)

Source: `results/eval/test_final/ft/predictions.jsonl`, with automatic tags in `errors_tagged.jsonl` (from `python -m kormat.eval.errors`).
Plain greedy decoding, 206 criteria from 8 test patients.

The model made **55 label errors** (accuracy 0.733). Another **33 criteria** got the right label but the wrong evidence set: in 19 the model cited sentences outside the expert set, and in 17 it missed expert sentences (some do both).
There were no INVALID outputs and no hallucinated sentence IDs.

## Error categories (label errors)
| category | gold → predicted | count | what it means |
|---|---|---|---|
| **A. Absence treated as proof** | exclusion: not enough information → not excluded | **28** | no sentence mentions the exclusion condition, and the model concludes "not excluded" instead of "not enough information" |
| B. Applicability collapsed | exclusion: not applicable → not excluded | 9 | the criterion cannot apply to this patient (e.g. a procedure they are not having), and the model answers "not excluded" |
| C. Over-abstention | inclusion: not included / included → not enough information; exclusion: not excluded → not enough information | 7 + 6 + 1 = 14 | the note does contain the deciding fact, but it needs a clinical inference (vitals → shock, "underwent X" → not a candidate for Y) |
| D. Applicability vs. missing info | inclusion: not applicable → not enough information; exclusion: not excluded → not applicable | 3 + 1 = 4 | sex-specific or conditional criteria ("if female …") |

Category A is a systematic failure. **All 28** exclusion criteria labelled "not enough information" were predicted "not excluded", so exclusion "not enough information" has F1 = 0 on test. The training split explains it: 292 of 388 exclusion criteria are "not excluded", and 218 of those 292 have **no** evidence sentence. Only 65 are "not enough information". The model learned the shortcut "nothing mentioned → not excluded", which is exactly what the brief warns against: absent evidence treated as proof of eligibility.
For a screening tool this is the dangerous direction, because it hides missing information instead of flagging it. Categories A and B together make up 37 of the 55 errors.

## Ten traced cases (≥ 8 required)
| # | id | type | criterion (abridged) | gold | pred | cited evidence (gold / pred) | diagnosis |
|---|---|---|---|---|---|---|---|
| 1 | 169 | excl | eGFR < 30 ml/min/1.73 m² | not enough information | not excluded | [] / [] | **A, missing information.** No renal function is reported; the model asserts the patient is not excluded with no evidence. |
| 2 | 174 | excl | previous axillary surgery | not enough information | not excluded | [] / [] | **A, missing information.** Surgical history is silent on the axilla; absence is treated as a negative finding. |
| 3 | 683 | excl | previous tonsillectomy | not applicable | not excluded | [] / [] | **B.** The experts judged the criterion not applicable to this patient; the model gives the default "not excluded". |
| 4 | 693 | excl | sleep disorder other than OSA | not applicable | not excluded | [] / [] | **B.** Same shortcut on a condition-specific exclusion. |
| 5 | 162 | incl | undergoing open AAA repair | not included | not enough information | [] / [0, 1, 2] | **C, contradictory information.** The note describes a femoral pseudo-aneurysm after catheterisation, a different vascular problem that contradicts the criterion. The model cites the right sentences but abstains instead of ruling the criterion out. |
| 6 | 161 | incl | undergoing elective carotid endarterectomy | not included | not enough information | [] / [] | **C.** The acute presentation implies the patient is not an elective endarterectomy candidate. The experts infer "not included"; the model abstains. |
| 7 | 185 | incl | diagnosed with shock or ARDS | included | not enough information | [2, 3] / [0, 1, 2, 3] | **C, implicit evidence.** BP 80/60 with tachycardia and tachypnoea is shock, but the note never says the word. The model finds the right sentences but will not make the clinical inference. |
| 8 | 176 | incl | no history of central venous catheterisation | included | not enough information | [0] / [0] | **C.** It cites the right sentence (femoral arterial catheterisation) but cannot tell arterial from central-venous access, so it abstains. |
| 9 | 422 | incl | if female, using birth control | not applicable | not enough information | [3] / [3] | **D.** "No sexual partners" makes contraception moot (experts: not applicable). The model reads it as missing information. Right evidence, wrong category. |
| 10 | 322 | excl | pregnant, or childbearing age without contraception | not excluded | not applicable | [0] / [] | **D.** The patient is a 48-year-old man. The experts say "not excluded" (citing sex/age); the model says "not applicable" and cites nothing. Both readings are defensible: an annotation-convention ambiguity. |
| 11 | 196 | excl | less than 18 years of age | not excluded | not enough information | [0] / [] | **C, unsupported abstention.** Sentence 0 states the age (40). The model misses it, a plain retrieval failure. |
| 12 | 307 | incl | liver-damage types before volumetry | not enough information | not enough information | [3] / [0–4] | **Evidence over-citation.** The label is right, but the model cites the whole note: high recall, low precision (macro evidence precision suffers). |

Cases 1, 2 and 11 are the **missing-information** failures. Case 5 is the **contradictory** case. Cases 7, 8, 9 and 12 show the model **localising the evidence correctly while misjudging it**. That is an important property: the explanation still points the user to the right place in the note.

## What would fix it (not run, for lack of time)
1. Re-balance training for exclusion criteria (oversample exclusion "not enough information" and "not applicable"), or add a loss weight on those label tokens.
2. A rule check in the app: an exclusion prediction of "not excluded" with **empty evidence** is downgraded to "not enough information / verify" and flagged in the missing-information list. This is a deterministic guard against category A that needs no retraining. *Not applied to the reported numbers.*
3. A larger base model for the clinical-inference cases (C), which need medical knowledge, not just reading.
