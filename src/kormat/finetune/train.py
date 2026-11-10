"""LoRA SFT trainer (completion-only loss). Device-agnostic: CUDA bf16 if available, else CPU fp32.

Written for this repo from the configuration of the shared adapter (see configs/train.yaml); a plain
PyTorch loop so every step is inspectable. Usage:
    python -m kormat.finetune.train --config configs/train.yaml
    python -m kormat.finetune.train --config configs/train_smoke.yaml [--mode smoke|overfit]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import resource
import time
from pathlib import Path

import torch
import yaml
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

from kormat.common.paths import ROOT
from kormat.common.runlog import set_seed
from kormat.data import load_split
from kormat.data.sft import build_sft, encode_example
from kormat.prompting.template import build_messages, render_prompt


def pick_device_dtype():
    if torch.cuda.is_available():
        return torch.device("cuda"), torch.bfloat16
    return torch.device("cpu"), torch.float32


def peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def collate_one(ex: dict, device):
    return {k: torch.tensor([ex[k]], device=device) for k in ("input_ids", "attention_mask", "labels")}


def build_examples(cfg: dict, tok) -> list[dict]:
    """Examples for training. `subset` (smoke/overfit) takes the shortest n of the first `pool` train records."""
    d = cfg["data"]
    if d.get("subset"):
        recs = load_split(d.get("split", "train"))[: int(d["pool"])]
        encoded = [e for e in (encode_example(r, tok, cfg["max_len"]) for r in recs) if e is not None]
        encoded.sort(key=lambda e: len(e["input_ids"]))
        return encoded[: int(d["subset"])]
    return build_sft(d.get("split", "train"), tok, cfg["max_len"])


def make_model(cfg: dict, device, dtype):
    model = AutoModelForCausalLM.from_pretrained(cfg["base_model"], torch_dtype=dtype)
    lc = cfg["lora"]
    model = get_peft_model(model, LoraConfig(r=lc["r"], lora_alpha=lc["alpha"], lora_dropout=lc["dropout"],
                                             target_modules=lc["target_modules"], task_type="CAUSAL_LM"))
    if cfg.get("grad_checkpointing", True):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    # LoRA params in fp32 even when base is bf16 (stable optimiser state)
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    return model.to(device)


@torch.no_grad()
def greedy_reproduce(model, tok, recs: list[dict], max_new: int = 48) -> list[dict]:
    from kormat.modeling import eos_ids

    model.eval()
    model.config.use_cache = True
    out = []
    for r in recs:
        prompt = render_prompt(tok, build_messages(r["sentences"], r["criterion_type"], r["criterion_text"]))
        ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
        gen = model.generate(ids, max_new_tokens=max_new, do_sample=False, repetition_penalty=1.0, temperature=None, top_p=None, top_k=None, eos_token_id=sorted(eos_ids(tok, model)),
                             pad_token_id=tok.pad_token_id)
        out.append(tok.decode(gen[0, ids.shape[1]:], skip_special_tokens=True).strip())
    return out


def train(cfg: dict, out_dir: Path, ckpt_dir: Path) -> dict:
    set_seed(cfg["seed"])
    device, dtype = pick_device_dtype()
    if cfg.get("threads"):
        torch.set_num_threads(int(cfg["threads"]))
    tok = AutoTokenizer.from_pretrained(cfg["base_model"])
    examples = build_examples(cfg, tok)
    model = make_model(cfg, device, dtype)
    model.train()
    accum = int(cfg["grad_accum"])
    n_micro_total = len(examples) * int(cfg["epochs"])
    total_steps = int(cfg.get("max_steps") or math.ceil(n_micro_total / accum))
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=float(cfg["lr"]), weight_decay=float(cfg.get("weight_decay", 0.0)))
    sched = get_cosine_schedule_with_warmup(opt, max(1, int(round(cfg["warmup_ratio"] * total_steps))), total_steps)
    out_dir.mkdir(parents=True, exist_ok=True)
    gen = torch.Generator().manual_seed(cfg["seed"])
    rows, step, micro = [], 0, 0
    order: list[int] = []
    t_start = time.perf_counter()
    while step < total_steps:
        losses, t0 = [], time.perf_counter()
        for _ in range(accum):
            if not order:
                order = torch.randperm(len(examples), generator=gen).tolist()
            ex = examples[order.pop()]
            loss = model(**collate_one(ex, device)).loss
            (loss / accum).backward()
            losses.append(loss.item())
            micro += 1
        torch.nn.utils.clip_grad_norm_(params, float(cfg.get("max_grad_norm", 1.0)))
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        step += 1
        rows.append({"step": step, "loss": sum(losses) / len(losses), "lr": sched.get_last_lr()[0],
                     "sec": round(time.perf_counter() - t0, 2), "peak_rss_mb": round(peak_rss_mb())})
        print(f"step {step}/{total_steps} loss {rows[-1]['loss']:.4f} lr {rows[-1]['lr']:.2e} {rows[-1]['sec']}s", flush=True)
    with (out_dir / "train_log.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, list(rows[0]))
        w.writeheader(); w.writerows(rows)
    model.save_pretrained(ckpt_dir)
    k = max(1, min(5, len(rows) // 3))
    summary = {
        "device": str(device), "dtype": str(dtype), "base_model": cfg["base_model"], "n_examples": len(examples),
        "example_token_len_min_max": [min(len(e["input_ids"]) for e in examples), max(len(e["input_ids"]) for e in examples)],
        "optimizer_steps": step, "micro_batches": micro, "grad_accum": accum,
        "loss_first": rows[0]["loss"], "loss_last": rows[-1]["loss"],
        "loss_mean_first_k": sum(r["loss"] for r in rows[:k]) / k, "loss_mean_last_k": sum(r["loss"] for r in rows[-k:]) / k, "k": k,
        "sec_per_step_mean": sum(r["sec"] for r in rows) / len(rows),
        "sec_per_microbatch_mean": (time.perf_counter() - t_start) / micro,
        "peak_rss_mb": round(peak_rss_mb()), "wall_s": round(time.perf_counter() - t_start, 1),
        "adapter_dir": str(ckpt_dir),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return {"model": model, "tok": tok, "summary": summary, "rows": rows, "examples": examples}


def plot_curve(rows: list[dict], path_png: Path, title: str, extra_pdf: Path | None = None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.plot([r["step"] for r in rows], [r["loss"] for r in rows], marker="o", ms=3)
    ax.set_xlabel("optimizer step"); ax.set_ylabel("completion-only loss"); ax.set_title(title); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(path_png, dpi=150)
    if extra_pdf:
        extra_pdf.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(extra_pdf)
    plt.close(fig)


def check_reload(cfg: dict, ckpt_dir: Path, tok, rec: dict) -> dict:
    """Reload the saved adapter on a fresh base and run one greedy generation."""
    base = AutoModelForCausalLM.from_pretrained(cfg["base_model"], torch_dtype=torch.float32)
    m = PeftModel.from_pretrained(base, str(ckpt_dir)).eval()
    txt = greedy_reproduce(m, tok, [rec], max_new=48)[0]
    return {"reload_ok": True, "sample_output": txt}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--mode", default="train", choices=["train", "smoke", "overfit"])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    cfg = yaml.safe_load(Path(a.config).read_text())
    if a.mode == "overfit":
        cfg = {**cfg, **cfg["overfit"]}
        cfg["data"] = {**cfg["data"], **cfg["overfit"].get("data", {})}
    out_dir = Path(a.out) if a.out else ROOT / cfg["out_dir"] / ("overfit" if a.mode == "overfit" else "")
    ckpt_dir = ROOT / cfg["adapter_out"] / ("overfit" if a.mode == "overfit" else "")
    res = train(cfg, out_dir, ckpt_dir)
    s = res["summary"]
    if a.mode != "train":
        plot_curve(res["rows"], out_dir / "loss_curve.png", f"{cfg['name']} ({cfg['base_model'].split('/')[-1]}, CPU)",
                   ROOT / "report" / "figures" / f"train_smoke_{a.mode}.pdf")
    recs = None
    if a.mode == "smoke":
        recs = load_split("train")[:1]
        s.update(check_reload(cfg, ckpt_dir, res["tok"], recs[0]))
    if a.mode == "overfit":
        tok = res["tok"]
        # recover the records matching the selected examples by re-encoding
        pool = load_split("train")[: int(cfg["data"]["pool"])]
        enc = [(r, encode_example(r, tok, cfg["max_len"])) for r in pool]
        enc = sorted([(r, e) for r, e in enc if e is not None], key=lambda t: len(t[1]["input_ids"]))[: int(cfg["data"]["subset"])]
        from kormat.prompting.template import format_target
        targets = [format_target(r["label"], r["evidence"]) for r, _ in enc]
        outs = greedy_reproduce(res["model"], tok, [r for r, _ in enc])
        exact = [o == t for o, t in zip(outs, targets)]
        s["reproduced_exact"] = sum(exact); s["n_checked"] = len(exact)
        s["pairs"] = [{"target": t, "output": o, "exact": e} for t, o, e in zip(targets, outs, exact)]
    (out_dir / "summary.json").write_text(json.dumps(s, indent=2))
    print(json.dumps({k: v for k, v in s.items() if k != "pairs"}, indent=2))


if __name__ == "__main__":
    main()
