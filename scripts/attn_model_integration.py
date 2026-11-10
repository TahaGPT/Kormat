"""Run a Qwen2 model with normal SDPA vs tiled_attention patched in: identical greedy tokens? max logit diff?
Usage: attn_model_integration.py [--tiny | --real]   (--real must run under flock .model.lock)"""
import json, sys

import torch

from kormat.attention.integration import tiled_sdpa_patch
from kormat.common.paths import RESULTS
from kormat.common.runlog import capture_env, set_seed

PROMPTS = ["The capital of France is", "def fibonacci(n):", "Attention is all you need because"]


@torch.inference_mode()
def compare(model, ids, new_tokens=12, block=16):
    out = {}
    base_logits = model(ids).logits
    base_gen = model.generate(ids, max_new_tokens=new_tokens, do_sample=False, pad_token_id=0)
    cnt = {}
    with tiled_sdpa_patch(block, block, "kv_blocked", cnt):
        t_logits = model(ids).logits
        t_gen = model.generate(ids, max_new_tokens=new_tokens, do_sample=False, pad_token_id=0)
    diff = (base_logits - t_logits).abs()
    return {"prompt_len": ids.shape[1], "max_logit_diff": diff.max().item(), "mean_logit_diff": diff.mean().item(),
            "logit_abs_max": base_logits.abs().max().item(), "greedy_tokens_identical": bool(torch.equal(base_gen, t_gen)),
            "new_tokens": base_gen[0, ids.shape[1]:].tolist(), "tiled_calls": cnt.get("calls", 0)}


def main():
    set_seed(0)
    if "--real" in sys.argv:
        from kormat.modeling import load_draft
        ld = load_draft(); model, tok = ld.model, ld.tokenizer
        res = {"model": ld.model_id, "cases": []}
        for p in PROMPTS:
            r = compare(model, tok(p, return_tensors="pt").input_ids); r["prompt"] = p
            r["text"] = tok.decode(r["new_tokens"]); res["cases"].append(r); print(r, flush=True)
        name = "model_integration.json"
    else:
        from kormat.modeling import tiny_qwen
        model = tiny_qwen(); res = {"model": "tiny_qwen (random)", "cases": []}
        for n in (5, 40):
            res["cases"].append(compare(model, torch.randint(1, 1000, (1, n)))); print(res["cases"][-1])
        name = "model_integration_tiny.json"
    res["env"] = capture_env(); res["block"] = 16; res["variant"] = "kv_blocked"
    (RESULTS / "attention" / name).write_text(json.dumps(res, indent=2, default=str))


if __name__ == "__main__":
    main()
