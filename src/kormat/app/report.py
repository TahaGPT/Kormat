"""Print-friendly HTML report (self-contained)."""
from __future__ import annotations

from html import escape

DISCLAIMER = "Educational screening prototype, not a clinical enrollment decision."
CHIP = {"included": "#1f9d55", "not excluded": "#1f9d55", "not included": "#d64545", "excluded": "#d64545",
        "not enough information": "#d98e04", "INVALID": "#d98e04", "not applicable": "#6b7280"}


def render_html(r: dict) -> str:
    cited = {int(i) for c in r["criteria"] for i in c["evidence_ids"]}
    note = "".join(
        f'<li value="{i}"{" class=hl" if int(i) in cited else ""}><span class=n>{i}</span> {escape(t)}</li>'
        for i, t in sorted(((int(k), v) for k, v in r["sentences"].items())))
    rows = ""
    for c in r["criteria"]:
        col = CHIP.get(c["label"], "#6b7280")
        ev = "<br>".join(f"[{i}] {escape(t)}" for i, t in c["evidence_text"].items()) or "<i>none cited</i>"
        rows += (f"<tr><td>{c['index'] + 1}</td><td>{escape(c['type'])}</td><td>{escape(c['text'])}</td>"
                 f"<td><b style='color:{col}'>{escape(c['label'])}</b></td><td>{ev}</td><td>{escape(c['explanation'])}</td></tr>")
    a = r["aggregate"]
    miss = "".join(f"<li>{escape(m['text'] or '')} <i>({m['type']})</i></li>" for m in a["missing_information"]) or "<li>None</li>"
    prov = "".join(f"<tr><th>{escape(str(k))}</th><td>{escape(str(v))}</td></tr>" for k, v in r["provenance"].items())
    return f"""<!doctype html><html lang=en><head><meta charset=utf-8><title>Kormat screening report {escape(r.get('id',''))}</title>
<style>body{{font:14px/1.5 system-ui,sans-serif;max-width:980px;margin:24px auto;padding:0 16px;color:#111}}
.disc{{border:2px solid #b45309;background:#fff7ed;padding:8px 12px;font-weight:600}}
table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #ccc;padding:4px 8px;vertical-align:top;text-align:left}}
li.hl{{background:#fde68a}}ol{{padding-left:0;list-style:none}}.n{{color:#6b7280;font-size:12px;margin-right:4px}}
.out{{font-size:20px;font-weight:700}}@media print{{body{{margin:0}}}}</style></head><body>
<div class=disc>{DISCLAIMER}</div><h1>Kormat eligibility screening report</h1>
<p>Report id {escape(r.get('id',''))} &middot; {escape(r['provenance'].get('timestamp',''))}</p>
<h2>Preliminary outcome</h2><p class=out>{escape(a['outcome'])}</p><p>{escape(a['reason'])}</p>
<p><b>Rule:</b> {escape(a['rule'])}</p>
<h2>Patient note (cited sentences highlighted)</h2><ol>{note}</ol>
<h2>Criterion-level results</h2><table><tr><th>#</th><th>Type</th><th>Criterion</th><th>Label</th><th>Evidence</th><th>Explanation</th></tr>{rows}</table>
<h2>Missing information</h2><ul>{miss}</ul>
<h2>Provenance</h2><table>{prov}</table>
<p class=disc>{DISCLAIMER}</p></body></html>"""
