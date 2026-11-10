"""Inference engine for the app: real (Qwen + LoRA) or fake (KORMAT_FAKE_ENGINE=1)."""
from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
from datetime import datetime, timezone

from kormat.prompting.labels import INVALID, LABELS


# ---------------------------------------------------------------- lazy deps with small fallbacks
def split_sentences(text: str) -> dict[int, str]:
    try:
        from kormat.data.sentences import ensure_sentences

        return ensure_sentences(text)
    except Exception:
        pass
    lines = [ln for ln in text.replace("\r", "").split("\n") if ln.strip()]
    numbered = [re.match(r"^\s*(\d+)\.\s+(.*)$", ln) for ln in lines]
    if lines and all(numbered):
        return {int(m.group(1)): m.group(2).strip() for m in numbered}
    parts: list[str] = []
    for ln in lines:
        parts += [p.strip() for p in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", ln) if p.strip()]
    return {i: p for i, p in enumerate(parts)}


def _parse(raw: str, ctype: str, valid_ids: set[int]) -> dict:
    try:
        from kormat.prompting.parse import parse_output

        return parse_output(raw, ctype, valid_ids)
    except ImportError:
        pass
    m = re.search(r"\{.*\}", raw, re.S)
    try:
        d = json.loads(m.group(0)) if m else {}
        label = d.get("label")
        ev = [int(i) for i in d.get("evidence", [])]
        if label not in LABELS[ctype]:
            return {"label": INVALID, "evidence": [], "status": "invalid_label", "hallucinated_ids": []}
        return {"label": label, "evidence": [i for i in ev if i in valid_ids], "status": "ok",
                "hallucinated_ids": [i for i in ev if i not in valid_ids]}
    except Exception:
        return {"label": INVALID, "evidence": [], "status": "unparseable", "hallucinated_ids": []}


def _explain(label, ev, sentences, ctype, ctext) -> str:
    try:
        from kormat.eval.explain import build_explanation

        return build_explanation(label, ev, sentences, ctype, ctext)
    except ImportError:
        if label == INVALID:
            return "The model's answer could not be parsed, so this criterion is treated as missing information."
        if ev:
            return f"Labelled \"{label}\" citing sentence(s) {', '.join(map(str, ev))}."
        return f"Labelled \"{label}\"; no supporting sentence was cited."


def provenance(engine_name: str, mode: str, gamma: int | None, max_new: int, extra: dict | None = None) -> dict:
    sha = "unknown"
    try:
        from kormat.common.runlog import git_sha

        sha = git_sha()
    except Exception:
        pass
    p = {"engine": engine_name, "git_sha": sha, "decoding_mode": mode,
         "gamma": gamma if mode == "speculative" else None, "max_new_tokens": max_new,
         "greedy": True, "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    p.update(extra or {})
    return p


def finish_result(raw: str, sentences, ctype, ctext, timing: dict, mode: str) -> dict:
    parsed = _parse(raw, ctype, set(sentences))
    label = parsed["label"]
    ev = [i for i in parsed.get("evidence", []) if i in sentences]
    return {
        "type": ctype, "text": ctext, "label": label, "status": parsed.get("status", "ok"),
        "evidence_ids": ev, "evidence_text": {str(i): sentences[i] for i in ev},
        "hallucinated_ids": parsed.get("hallucinated_ids", []),
        "explanation": _explain(label, ev, sentences, ctype, ctext),
        "missing_information": label in ("not enough information", INVALID),
        "raw_output": raw, "mode": mode, "timing": timing,
    }


# ---------------------------------------------------------------- fake engine
class FakeEngine:
    name = "fake"

    def __init__(self):
        self.lock = threading.Lock()

    def state(self) -> dict:
        return {"status": "ready", "engine": "fake", "detail": "deterministic fake engine (no model loaded)"}

    def start_loading(self):
        pass

    def ready(self) -> bool:
        return True

    def info(self) -> dict:
        return {"model_id": "fake-engine", "adapter": None, "dtype": "none"}

    def _decide(self, sentences, ctype, ctext):
        h = int(hashlib.sha256(ctext.encode()).hexdigest(), 16)
        labels = LABELS[ctype]
        label = labels[h % len(labels)]
        words = {w for w in re.findall(r"[a-z]{4,}", ctext.lower())}
        scored = sorted(sentences, key=lambda i: -len(words & set(re.findall(r"[a-z]{4,}", sentences[i].lower()))))
        ev = [scored[0]] if scored and label not in ("not applicable", "not enough information") else []
        return {"label": label, "evidence": ev}

    def run_decode(self, sentences, ctype, ctext, mode, gamma, max_new, tokens_only=False) -> dict:
        d = self._decide(sentences, ctype, ctext)
        raw = json.dumps(d)
        n = max(8, len(raw) // 3)
        base = {"no_cache": 3.0, "kv_cached": 1.0, "speculative": 0.7, "hf_generate": 0.9}[mode]
        t = {"prefill_ms": 40.0 * base, "decode_ms": 12.0 * n * base, "n_prompt": 300, "n_new": n,
             "tokens_per_s": 1000 / (12.0 * base), "peak_rss_mb": 7000.0 + (80 if mode == "speculative" else 0),
             "kv_cache_mb": 0.0 if mode == "no_cache" else 9.5, "stats": None}
        if mode == "speculative":
            t["stats"] = {"gamma": gamma, "acceptance_rate": 0.71, "mean_accepted_per_iter": gamma * 0.71}
        return {"raw": raw, "tokens": [ord(c) for c in raw], "timing": t}

    def screen_one(self, sentences, ctype, ctext, mode="kv_cached", gamma=4, max_new=64) -> dict:
        with self.lock:
            r = self.run_decode(sentences, ctype, ctext, mode, gamma, max_new)
        return finish_result(r["raw"], sentences, ctype, ctext, r["timing"], mode)

    def lab(self, sentences, ctype, ctext, modes, gamma, max_new) -> list[dict]:
        with self.lock:
            out = [dict(mode=m, **self.run_decode(sentences, ctype, ctext, m, gamma, max_new)) for m in modes]
        return out

    def attention(self, length: int) -> dict:
        return {"L": length, "note": "fake engine: no measurement", "available": False}


# ---------------------------------------------------------------- real engine
class RealEngine:
    name = "qwen"

    def __init__(self):
        self.lock = threading.Lock()
        self._state = {"status": "idle", "detail": "not loaded yet"}
        self.target = self.draft = None
        self._thr: threading.Thread | None = None

    def state(self) -> dict:
        return {"engine": "qwen", **self._state}

    def ready(self) -> bool:
        return self._state["status"] == "ready"

    def start_loading(self):
        if self._thr and self._thr.is_alive() or self.ready():
            return
        self._thr = threading.Thread(target=self._load, daemon=True)
        self._thr.start()

    def _load(self):
        try:
            from kormat.modeling import load_draft, load_target

            self._state = {"status": "loading", "detail": "loading target model (1.5B + LoRA, merged)...", "progress": 0.1}
            self.target = load_target(finetuned=True, merge=True)
            self._state = {"status": "loading", "detail": "loading draft model (0.5B)...", "progress": 0.7}
            self.draft = load_draft()
            self._state = {"status": "ready", "detail": "models loaded", "progress": 1.0}
        except Exception as e:  # noqa: BLE001
            self._state = {"status": "error", "detail": f"{type(e).__name__}: {e}"}

    def info(self) -> dict:
        t = self.target
        if not t:
            return {}
        return {"model_id": t.model_id, "adapter": t.adapter, "dtype": str(t.dtype).replace("torch.", "")}

    def run_decode(self, sentences, ctype, ctext, mode, gamma, max_new) -> dict:
        import torch
        from kormat.decoding import api
        from kormat.modeling import eos_ids
        from kormat.prompting.template import build_messages, render_prompt

        tok, model = self.target.tokenizer, self.target.model
        prompt = render_prompt(tok, build_messages(sentences, ctype, ctext))
        ids = tok(prompt, return_tensors="pt").input_ids
        eos = eos_ids(tok, model)
        with torch.inference_mode():
            if mode == "no_cache":
                r = api.greedy_nocache(model, ids, max_new, eos, tokenizer=tok)
            elif mode == "kv_cached":
                r = api.greedy_kv(model, ids, max_new, eos, tokenizer=tok)
            elif mode == "speculative":
                r = api.speculative_greedy(model, self.draft.model, ids, max_new, eos, gamma=gamma, tokenizer=tok)
            else:
                r = api.hf_generate(model, ids, max_new, eos, tokenizer=tok)
        stats = getattr(r, "stats", None)
        if stats:
            stats = {k: v for k, v in dict(stats).items() if k != "per_iter"} | {"per_iter": list(stats.get("per_iter", []))[:50]}
        timing = {k: getattr(r, k, None) for k in ("prefill_ms", "decode_ms", "n_prompt", "n_new", "tokens_per_s",
                                                  "peak_rss_mb", "kv_cache_mb")}
        timing["stats"] = stats
        text = r.text if getattr(r, "text", None) else tok.decode(r.tokens, skip_special_tokens=True)
        return {"raw": text, "tokens": [int(t) for t in r.tokens], "timing": timing}

    def screen_one(self, sentences, ctype, ctext, mode="kv_cached", gamma=4, max_new=64) -> dict:
        with self.lock:
            r = self.run_decode(sentences, ctype, ctext, mode, gamma, max_new)
        return finish_result(r["raw"], sentences, ctype, ctext, r["timing"], mode)

    def lab(self, sentences, ctype, ctext, modes, gamma, max_new) -> list[dict]:
        with self.lock:
            return [dict(mode=m, **self.run_decode(sentences, ctype, ctext, m, gamma, max_new)) for m in modes]

    def attention(self, length: int) -> dict:
        from kormat.attention.benchmark import run_micro

        with self.lock:
            return {"available": True, "result": run_micro(length)}


def make_engine():
    if os.environ.get("KORMAT_FAKE_ENGINE") == "1":
        return FakeEngine()
    return RealEngine()
