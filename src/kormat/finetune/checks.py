"""Model-freeze checks (run under the model lock): merged-vs-unmerged equivalence and prompt-contract compliance."""
from __future__ import annotations

import json

import torch
from kormat.finetune.adapter import attach_adapter
from transformers import AutoModelForCausalLM, AutoTokenizer

from kormat.common.paths import RESULTS
from kormat.data import load_split
from kormat.modeling import ADAPTER_DIR, TARGET_ID, eos_ids
from kormat.prompting.template import build_messages, render_prompt

OUT = RESULTS / "finetune"


def five_train(n=5):
    recs = [r for r in load_split("train") if r["criterion_text"].strip()]
    # shortest of the first 60 keeps the locked CPU run short; distinct labels not required
    recs = sorted(recs[:60], key=lambda r: sum(len(t) for t in r["sentences"].values()))
    return recs[:n]


def prompt_ids(tok, r):
    p = render_prompt(tok, build_messages(r["sentences"], r["criterion_type"], r["criterion_text"]))
    return tok(p, return_tensors="pt", add_special_tokens=False).input_ids


@torch.inference_mode()
def greedy(model, tok, ids, n=48):
    g = model.generate(ids, max_new_tokens=n, do_sample=False, repetition_penalty=1.0, temperature=None, top_p=None, top_k=None, eos_token_id=sorted(eos_ids(tok, model)),
                       pad_token_id=tok.pad_token_id)
    return g[0, ids.shape[1]:].tolist()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(TARGET_ID)
    recs = five_train()
    base = AutoModelForCausalLM.from_pretrained(TARGET_ID, torch_dtype=torch.float32)
    peft = attach_adapter(base, ADAPTER_DIR).eval()
    items = []
    with torch.inference_mode():
        un_logits = [peft(input_ids=prompt_ids(tok, r)).logits[0, -1].clone() for r in recs]
        un_out = [greedy(peft, tok, prompt_ids(tok, r)) for r in recs]
    # contract check on the (unmerged) adapter outputs
    ok_all = True
    for r, o in zip(recs, un_out):
        txt = tok.decode(o, skip_special_tokens=True).strip()
        try:
            d = json.loads(txt)
            valid = set(d) == {"label", "evidence"} and isinstance(d["label"], str) and isinstance(d["evidence"], list) \
                and all(isinstance(i, int) for i in d["evidence"])
        except Exception:
            d, valid = None, False
        ok_all &= valid
        items.append({"annotation_id": r["annotation_id"], "criterion_type": r["criterion_type"], "output": txt, "valid_json_schema": valid,
                      "label_in_allowed_set": bool(valid and d["label"] in __import__("kormat.prompting.labels", fromlist=["LABELS"]).LABELS[r["criterion_type"]]),
                      "gold_label": r["label"], "gold_evidence": r["evidence"]})
    (OUT / "prompt_contract_check.json").write_text(json.dumps(
        {"model": "base + shared adapter (unmerged), fp32, greedy, max_new_tokens=48", "n": len(items), "all_valid_json": ok_all,
         "n_valid": sum(i["valid_json_schema"] for i in items), "n_label_allowed": sum(i["label_in_allowed_set"] for i in items),
         "examples": items}, indent=2))
    merged = peft.merge_and_unload().eval()
    with torch.inference_mode():
        m_logits = [merged(input_ids=prompt_ids(tok, r)).logits[0, -1] for r in recs]
        m_out = [greedy(merged, tok, prompt_ids(tok, r)) for r in recs]
    diffs = [float((a - b).abs().max()) for a, b in zip(un_logits, m_logits)]
    same = [a == b for a, b in zip(un_out, m_out)]
    (OUT / "merge_check.json").write_text(json.dumps(
        {"n_prompts": len(recs), "annotation_ids": [r["annotation_id"] for r in recs], "greedy_outputs_identical": all(same),
         "per_prompt_identical": same, "max_abs_logit_diff_last_pos": diffs, "max_abs_logit_diff": max(diffs),
         "unmerged_outputs": [tok.decode(o, skip_special_tokens=True) for o in un_out], "dtype": "float32"}, indent=2))
    print(open(OUT / "merge_check.json").read()[:1500]); print(open(OUT / "prompt_contract_check.json").read()[:1800])


if __name__ == "__main__":
    main()
