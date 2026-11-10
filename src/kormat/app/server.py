"""FastAPI backend for the Kormat screening app."""
from __future__ import annotations

import json
import re
import time
import urllib.request
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from kormat.app import engine as eng
from kormat.app.aggregate import RULE_TEXT, aggregate
from kormat.app.criteria_parse import parse_criteria
from kormat.app.report import render_html
from kormat.app.schemas import (CriterionIn, MAX_CRITERIA, MAX_CRITERION_CHARS, MAX_NOTE_CHARS, MODES, AttentionRequest,
                                LabRequest, ParseRequest, ScreenRequest)
from kormat.app.store import Store
from kormat.common.paths import RESULTS, ROOT

STATIC = Path(__file__).parent / "static"


class AppError(Exception):
    def __init__(self, status: int, code: str, message: str, hint: str | None = None):
        self.status, self.code, self.message, self.hint = status, code, message, hint


def _err(status, code, message, hint=None, headers=None):
    return JSONResponse({"error": {"code": code, "message": message, "hint": hint}}, status_code=status, headers=headers)


def validate_note(note: str) -> str:
    if not note or not note.strip():
        raise AppError(400, "empty_note", "The patient note is empty. Paste a note or load a sample.")
    if len(note) > MAX_NOTE_CHARS:
        raise AppError(400, "note_too_long", f"The note has {len(note)} characters; the limit is {MAX_NOTE_CHARS}. Shorten it.")
    return note


def validate_criterion(c, i=None) -> tuple[str, str]:
    where = f"Criterion {i + 1}: " if i is not None else ""
    if c.type not in ("inclusion", "exclusion"):
        raise AppError(400, "bad_type", f"{where}type must be 'inclusion' or 'exclusion', got {c.type!r}.")
    if not c.text.strip():
        raise AppError(400, "empty_criterion", f"{where}criterion text is empty.")
    if len(c.text) > MAX_CRITERION_CHARS:
        raise AppError(400, "criterion_too_long", f"{where}criterion is longer than {MAX_CRITERION_CHARS} characters.")
    return c.type, c.text.strip()


def _latest_run(area: str):
    d = RESULTS / area
    if not d.exists():
        return None
    runs = sorted([p for p in d.iterdir() if p.is_dir()], key=lambda p: p.name)
    for run in reversed(runs):
        for fn in ("summary.json", "metrics.json"):
            f = run / fn
            if f.exists():
                try:
                    return {"run_id": f"{area}/{run.name}", "file": fn, "data": json.loads(f.read_text())}
                except Exception:
                    continue
    for fn in ("summary.json", "metrics.json"):
        f = d / fn
        if f.exists():
            try:
                return {"run_id": f"{area}/(root)", "file": fn, "data": json.loads(f.read_text())}
            except Exception:
                pass
    return None


BUILTIN_SAMPLE = {
    "id": "demo-1", "split": "demo",
    "note": ("1. 58-year-old woman with type 2 diabetes for 10 years.\n2. HbA1c was 8.4% last month.\n"
             "3. She takes metformin 1000 mg twice daily.\n4. She has no history of heart failure.\n"
             "5. Blood pressure is 128/80 mmHg.\n6. She is not pregnant and has had a hysterectomy."),
    "criteria": [
        {"type": "inclusion", "text": "Adults aged 18 years or older with type 2 diabetes"},
        {"type": "inclusion", "text": "HbA1c between 7.0% and 10.0%"},
        {"type": "exclusion", "text": "History of heart failure"},
        {"type": "exclusion", "text": "Current insulin therapy"},
        {"type": "exclusion", "text": "Pregnant or breastfeeding"},
    ],
}


def create_app(engine=None, db_path=None) -> FastAPI:
    app = FastAPI(title="Kormat", version="1.0")
    app.state.engine = engine or eng.make_engine()
    app.state.store = Store(db_path) if db_path else Store()

    _cache: dict = {}

    @app.on_event("startup")
    def _startup():
        import os
        if os.environ.get("KORMAT_LAZY") != "1":
            app.state.engine.start_loading()

    @app.exception_handler(AppError)
    async def _app_err(_: Request, e: AppError):
        h = {"Retry-After": "10"} if e.status == 503 else None
        return _err(e.status, e.code, e.message, e.hint, h)

    @app.exception_handler(RequestValidationError)
    async def _val_err(_: Request, e: RequestValidationError):
        msg = "; ".join(f"{'.'.join(str(x) for x in er['loc'][1:])}: {er['msg']}" for er in e.errors())
        return _err(422, "invalid_request", f"Request is malformed ({msg}).")

    @app.exception_handler(Exception)
    async def _any_err(_: Request, e: Exception):
        return _err(500, "internal_error", f"Unexpected error: {type(e).__name__}: {e}")

    def need_ready():
        en = app.state.engine
        if not en.ready():
            st = en.state()
            if st["status"] == "error":
                raise AppError(500, "model_error", f"The model failed to load: {st['detail']}")
            en.start_loading()
            raise AppError(503, "model_loading", "The model is still loading.", "Retry in about 10-30 seconds; see the status badge.")

    def check_mode(mode, gamma, max_new):
        if mode not in MODES:
            raise AppError(400, "unknown_mode", f"Unknown decoding mode {mode!r}. Choose one of: {', '.join(MODES)}.")
        if not 1 <= gamma <= 8:
            raise AppError(400, "bad_gamma", "gamma (draft tokens per step) must be between 1 and 8.")
        if not 8 <= max_new <= 128:
            raise AppError(400, "bad_max_new", "max_new_tokens must be between 8 and 128.")

    @app.get("/api/health")
    def health():
        st = app.state.engine.state()
        return {"app": "kormat", **st, "limits": {"max_note_chars": MAX_NOTE_CHARS, "max_criteria": MAX_CRITERIA,
                                                    "max_criterion_chars": MAX_CRITERION_CHARS},
                "modes": list(MODES), "rule": RULE_TEXT}

    @app.get("/api/samples")
    def samples():
        if "samples" in _cache:
            return _cache["samples"]
        out = []
        try:
            from kormat.data import load_split  # train/val only: the test split is sealed

            for split in ("val", "train"):
                try:
                    rows = load_split(split)
                except Exception:
                    continue
                out += _samples_from(rows, split)
                if len(out) >= 6:
                    break
        except Exception:
            pass
        _cache["samples"] = {"samples": [BUILTIN_SAMPLE] + out[:6]}
        return _cache["samples"]

    def _samples_from(rows, split):
        res: dict[str, dict] = {}
        try:
            for r in list(rows)[:400]:
                r = dict(r) if not isinstance(r, dict) else r
                pid = str(r.get("patient_id") or r.get("note_id") or r.get("id") or len(res))
                sents = r.get("sentences") or {}
                if not sents:
                    continue
                note = "\n".join(f"{int(k)}. {v}" for k, v in sorted(((int(k), v) for k, v in dict(sents).items())))
                s = res.setdefault(pid, {"id": f"{split}-{pid}", "split": split, "note": note, "criteria": []})
                ct, tx = r.get("criterion_type"), r.get("criterion_text") or r.get("criterion")
                if ct in ("inclusion", "exclusion") and tx and len(s["criteria"]) < 6:
                    s["criteria"].append({"type": ct, "text": tx})
        except Exception:
            return []
        return [s for s in res.values() if s["criteria"]][:5]

    @app.get("/api/stored")
    def stored():
        out = {}
        for area in ("kvcache", "speculative", "attention"):
            out[area] = _latest_run(area) or {"available": False, "message": "not available yet"}
        return out

    @app.post("/api/parse_criteria")
    def parse_ep(req: ParseRequest):
        crit = parse_criteria(req.block)
        if not crit:
            raise AppError(400, "no_criteria", "No criteria found. Use 'Inclusion Criteria:' / 'Exclusion Criteria:' headers with bullets or numbers.")
        return {"criteria": crit, "count": len(crit)}

    @app.get("/api/nct/{nct}")
    def nct(nct: str):
        nct = nct.strip().upper()
        if not re.fullmatch(r"NCT\d{8}", nct):
            raise AppError(400, "bad_nct", "NCT IDs look like NCT01234567 (NCT followed by 8 digits).")
        try:
            with urllib.request.urlopen(f"https://clinicaltrials.gov/api/v2/studies/{nct}", timeout=8) as r:
                d = json.load(r)
            text = d["protocolSection"]["eligibilityModule"]["eligibilityCriteria"]
        except Exception as e:  # noqa: BLE001
            raise AppError(502, "nct_unavailable", f"Could not fetch {nct} from ClinicalTrials.gov ({type(e).__name__}). "
                           "You may be offline; paste the criteria instead.")
        crit = parse_criteria(text)
        return {"nct": nct, "criteria": crit, "block": text}

    @app.post("/api/screen")
    def screen(req: ScreenRequest):
        note = validate_note(req.note)
        crit = list(req.criteria)
        if not crit and req.criteria_block:
            crit = [CriterionIn(**c)
                    for c in parse_criteria(req.criteria_block)]
        if not crit:
            raise AppError(400, "no_criteria", "Add at least one eligibility criterion.")
        if len(crit) > MAX_CRITERIA:
            raise AppError(400, "too_many_criteria", f"{len(crit)} criteria given; the limit is {MAX_CRITERIA}.")
        checked = [validate_criterion(c, i) for i, c in enumerate(crit)]
        check_mode(req.mode, req.gamma, req.max_new_tokens)
        sentences = eng.split_sentences(note)
        if not sentences:
            raise AppError(400, "empty_note", "No sentences could be extracted from the note.")
        need_ready()
        en = app.state.engine
        t0 = time.perf_counter()
        results = []
        for i, (ct, tx) in enumerate(checked):
            r = en.screen_one(sentences, ct, tx, req.mode, req.gamma, req.max_new_tokens)
            r["index"] = i
            results.append(r)
        agg = aggregate(results)
        prov = eng.provenance(en.name, req.mode, req.gamma, req.max_new_tokens,
                              {**en.info(), "total_ms": round((time.perf_counter() - t0) * 1000, 1)})
        report = {"note_text": note, "sentences": {str(k): v for k, v in sentences.items()},
                  "criteria": results, "aggregate": agg, "provenance": prov,
                  "disclaimer": "Educational screening prototype, not a clinical enrollment decision."}
        if req.save:
            app.state.store.save(report)
        return report

    @app.post("/api/lab")
    def lab(req: LabRequest):
        note = validate_note(req.note)
        if req.criterion is None:
            raise AppError(400, "no_criteria", "Choose or enter one criterion for the lab.")
        ct, tx = validate_criterion(req.criterion)
        if not req.modes:
            raise AppError(400, "no_modes", "Select at least one decoding mode.")
        for m in req.modes:
            check_mode(m, req.gamma, req.max_new_tokens)
        modes = list(dict.fromkeys(req.modes))
        sentences = eng.split_sentences(note)
        need_ready()
        runs = app.state.engine.lab(sentences, ct, tx, modes, req.gamma, req.max_new_tokens)
        ref = next((r for r in runs if r["mode"] == "kv_cached"), None)
        if ref is None:
            ref = app.state.engine.lab(sentences, ct, tx, ["kv_cached"], req.gamma, req.max_new_tokens)[0]
        out = []
        for r in runs:
            tokens = r["tokens"]
            out.append({**{k: v for k, v in r.items() if k != "tokens"}, "equal_to_kv_cached": tokens == ref["tokens"], "n_tokens": len(tokens)})
        return {"criterion": {"type": ct, "text": tx}, "reference": "kv_cached", "runs": out,
                "provenance": eng.provenance(app.state.engine.name, ",".join(modes), req.gamma, req.max_new_tokens,
                                             app.state.engine.info())}

    @app.post("/api/attention")
    def attention(req: AttentionRequest):
        if not 16 <= req.length <= 2048:
            raise AppError(400, "bad_length", "Sequence length must be between 16 and 2048.")
        try:
            return app.state.engine.attention(req.length)
        except ImportError:
            raise AppError(501, "not_available", "The attention benchmark is not available yet.")

    @app.get("/api/history")
    def history():
        return {"reports": app.state.store.list()}

    def _get(rid):
        r = app.state.store.get(rid)
        if not r:
            raise AppError(404, "not_found", f"No saved report with id {rid}.")
        return r

    @app.get("/api/history/{rid}")
    def history_one(rid: str):
        return _get(rid)

    @app.delete("/api/history/{rid}")
    def history_del(rid: str):
        if not app.state.store.delete(rid):
            raise AppError(404, "not_found", f"No saved report with id {rid}.")
        return {"deleted": rid}

    @app.get("/api/report/{rid}.html")
    def report_html(rid: str):
        return HTMLResponse(render_html(_get(rid)))

    @app.get("/api/report/{rid}.json")
    def report_json(rid: str):
        return Response(json.dumps(_get(rid), indent=2), media_type="application/json",
                        headers={"Content-Disposition": f'attachment; filename="kormat-{rid}.json"'})

    @app.post("/api/report.html")
    def report_adhoc(report: dict):
        try:
            return HTMLResponse(render_html(report))
        except Exception:
            raise AppError(400, "bad_report", "That report could not be rendered.")

    @app.get("/logo.png")
    def logo():
        return FileResponse(ROOT / "assets" / "logo.png")

    app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")
    return app


app = create_app()
