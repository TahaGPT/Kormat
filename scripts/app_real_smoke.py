"""One real-model smoke run of the app engine (run under flock). Writes results/app/<stamp>/."""
import json, time
from kormat.app.engine import RealEngine
from kormat.app.aggregate import aggregate
from kormat.common.paths import RESULTS
from kormat.common.runlog import git_sha
from kormat.data import load_split

rec = load_split("val")
pid = rec[0]["patient_id"]
rows = [r for r in rec if r["patient_id"] == pid][:4]
sents = rows[0]["sentences"]
e = RealEngine(); t0 = time.time(); e._load()
assert e.ready(), e.state()
out = {"load_s": round(time.time() - t0, 1), "info": e.info(), "criteria": []}
for r in rows:
    res = e.screen_one(sents, r["criterion_type"], r["criterion_text"], "kv_cached")
    res["gold_label_val"] = r["label"]; out["criteria"].append(res)
out["aggregate"] = aggregate(out["criteria"])
lab = e.lab(sents, rows[0]["criterion_type"], rows[0]["criterion_text"], ["no_cache", "kv_cached", "speculative", "hf_generate"], 4, 64)
ref = next(x for x in lab if x["mode"] == "kv_cached")["tokens"]
out["lab"] = [{"mode": x["mode"], "equal_to_kv": x["tokens"] == ref, "timing": x["timing"]} for x in lab]
d = RESULTS / "app" / f"{time.strftime('%Y%m%d-%H%M%S')}_{git_sha()}"; d.mkdir(parents=True, exist_ok=True)
(d / "real_smoke.json").write_text(json.dumps(out, indent=2, default=str)); print(d)
