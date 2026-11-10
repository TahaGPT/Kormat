"use strict";
const $ = (s, r = document) => r.querySelector(s);
const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const CHIP = { "included": "c-ok", "not excluded": "c-ok", "not included": "c-bad", "excluded": "c-bad", "not enough information": "c-warn", "INVALID": "c-warn", "not applicable": "c-na" };
const chip = l => `<span class="chip ${CHIP[l] || "c-na"}">${esc(l)}</span>`;
let crits = [], last = null, charts = {}, limits = { max_note_chars: 12000, max_criteria: 40 };

async function api(path, opts) {
  let r;
  try { r = await fetch(path, opts); } catch (e) { throw new Error("Cannot reach the server. Is it still running?"); }
  let d = null; try { d = await r.json(); } catch (e) { /* non-json */ }
  if (!r.ok) {
    const er = d && d.error ? d.error : { message: `Request failed (${r.status})` };
    const x = new Error(er.message); x.hint = er.hint; x.status = r.status; throw x;
  }
  return d;
}
const post = (p, body) => api(p, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
const showErr = (el, e) => { el.innerHTML = e ? `<div class="err" role="alert"><b>Problem:</b> ${esc(e.message)}${e.hint ? `<br><span class="muted">${esc(e.hint)}</span>` : ""}</div>` : ""; };

// theme + tabs
const root = document.documentElement;
try { root.dataset.theme = localStorage.getItem("kormat-theme") || "dark"; } catch (e) { root.dataset.theme = "dark"; }
const setThemeLabel = () => $("#theme").textContent = root.dataset.theme === "dark" ? "Light mode" : "Dark mode";
setThemeLabel();
$("#theme").onclick = () => { root.dataset.theme = root.dataset.theme === "dark" ? "light" : "dark"; try { localStorage.setItem("kormat-theme", root.dataset.theme); } catch (e) { } setThemeLabel(); };
document.querySelectorAll("nav button").forEach(b => b.onclick = () => {
  document.querySelectorAll("nav button").forEach(x => x.setAttribute("aria-selected", x === b));
  ["screen", "lab", "history", "about"].forEach(t => $("#tab-" + t).hidden = t !== b.dataset.tab);
  if (b.dataset.tab === "history") loadHistory();
  if (b.dataset.tab === "lab") { fillLabCrit(); loadStored(); }
});

// health
async function health() {
  const el = $("#health");
  try {
    const h = await api("/api/health"); limits = h.limits; $("#notemax").textContent = limits.max_note_chars;
    const pct = h.progress ? ` ${Math.round(h.progress * 100)}%` : "";
    el.className = "badge " + h.status; el.textContent = h.status === "ready" ? `model ready (${h.engine})` : `${h.status}${pct}: ${h.detail}`;
    $("#rulelist").innerHTML = h.rule.split(/\s(?=\d\.\s)/).map(s => `<li>${esc(s.replace(/^\d\.\s*/, ""))}</li>`).join("");
    if (h.status === "ready") return;
  } catch (e) { el.className = "badge error"; el.textContent = "server unreachable"; }
  setTimeout(health, 3000);
}
health();

// samples
let samples = [];
api("/api/samples").then(d => {
  samples = d.samples;
  samples.forEach((s, i) => $("#sample").insertAdjacentHTML("beforeend", `<option value="${i}">${esc(s.id)} (${esc(s.split)})</option>`));
}).catch(() => { });
$("#sample").onchange = e => {
  const s = samples[e.target.value]; if (!s) return;
  $("#note").value = s.note; crits = s.criteria.map(c => ({ ...c })); renderCrits(); countNote();
};
const countNote = () => { const n = $("#note").value.length; $("#notecount").textContent = n; $("#notecount").style.color = n > limits.max_note_chars ? "var(--bad)" : ""; };
$("#note").oninput = countNote;

// criteria rows
function renderCrits() {
  $("#crits").innerHTML = crits.map((c, i) => `<div class="crit"><select aria-label="Criterion ${i + 1} type" data-i="${i}" data-f="type">
   <option ${c.type === "inclusion" ? "selected" : ""}>inclusion</option><option ${c.type === "exclusion" ? "selected" : ""}>exclusion</option></select>
   <input aria-label="Criterion ${i + 1} text" data-i="${i}" data-f="text" value="${esc(c.text)}"><button class="btn" data-del="${i}" aria-label="Remove criterion ${i + 1}">x</button></div>`).join("");
  $("#critcount").textContent = `${crits.length} / ${limits.max_criteria}`;
}
$("#crits").addEventListener("input", e => { const t = e.target; if (t.dataset.f) crits[t.dataset.i][t.dataset.f] = t.value; });
$("#crits").addEventListener("click", e => { const d = e.target.dataset.del; if (d !== undefined) { crits.splice(+d, 1); renderCrits(); } });
$("#addrow").onclick = () => { crits.push({ type: "inclusion", text: "" }); renderCrits(); };
$("#clearrows").onclick = () => { crits = []; renderCrits(); };
$("#parse").onclick = async () => {
  try { showErr($("#screenerr")); const d = await post("/api/parse_criteria", { block: $("#block").value }); crits = crits.concat(d.criteria); renderCrits(); }
  catch (e) { showErr($("#screenerr"), e); }
};
$("#fetchnct").onclick = async () => {
  try { showErr($("#screenerr")); const d = await api("/api/nct/" + encodeURIComponent($("#nct").value.trim())); $("#block").value = d.block; crits = d.criteria; renderCrits(); }
  catch (e) { showErr($("#screenerr"), e); }
};
renderCrits();

// screening
$("#run").onclick = async () => {
  const b = $("#run"); b.disabled = true; $("#runstat").innerHTML = '<span class="spinner"></span> screening...'; showErr($("#screenerr"));
  try {
    last = await post("/api/screen", { note: $("#note").value, criteria: crits, mode: $("#mode").value, gamma: +$("#gamma").value || 4 });
    renderResults(last, 0);
    $("#runstat").textContent = `done in ${(last.provenance.total_ms / 1000).toFixed(1)} s`;
  } catch (e) { showErr($("#screenerr"), e); $("#runstat").textContent = ""; if (e.status === 503) setTimeout(health, 100); }
  b.disabled = false;
};
function highlight(ids) {
  document.querySelectorAll("#notelist li").forEach(li => li.classList.toggle("hl", ids.includes(+li.dataset.id)));
}
function renderResults(r, sel) {
  const a = r.aggregate, cls = a.outcome.startsWith("Likely") ? "c-bad" : a.outcome.startsWith("Undet") ? "c-warn" : "c-ok";
  const rows = r.criteria.map((c, i) => `<tr tabindex="0" data-i="${i}" class="${i === sel ? "sel" : ""}" aria-label="Criterion ${i + 1}, ${esc(c.label)}">
   <td>${i + 1}</td><td>${esc(c.type)}</td><td>${esc(c.text)}</td><td>${chip(c.label)}</td><td>${c.evidence_ids.map(x => "[" + x + "]").join(" ") || '<span class="muted">none</span>'}</td></tr>`).join("");
  const miss = a.missing_information.map(m => `<li>${esc(m.text)} <span class="muted">(${esc(m.type)})</span></li>`).join("") || "<li class='muted'>None</li>";
  const na = a.ignored_not_applicable.length ? `<p class="small muted">Not applicable (ignored in the decision): ${a.ignored_not_applicable.map(m => esc(m.text)).join("; ")}</p>` : "";
  const note = Object.entries(r.sentences).map(([k, v]) => `<li data-id="${k}" id="s${k}"><span class="n">${k}</span>${esc(v)}</li>`).join("");
  const prov = Object.entries(r.provenance).map(([k, v]) => `<dt>${esc(k)}</dt><dd>${esc(v ?? "n/a")}</dd>`).join("");
  $("#results").innerHTML = `
  <div class="card outcome"><div class="muted small">Preliminary outcome (screening aid, not ground truth)</div>
   <div class="big ${cls}">${esc(a.outcome)}</div><p>${esc(a.reason)}</p>
   <details><summary>How this was decided</summary><p class="small">${esc(a.rule)}</p></details>
   <div class="row no-print"><a class="btn" href="/api/report/${r.id}.html" target="_blank" rel="noopener">Open report (print / PDF)</a>
   <a class="btn" href="/api/report/${r.id}.json" download>Download JSON</a></div></div>
  <div class="card"><h2>Criterion results</h2><p class="small muted">Select a row (click or Enter) to highlight the cited sentences.</p>
   <div style="overflow-x:auto"><table><thead><tr><th>#</th><th>Type</th><th>Criterion</th><th>Label</th><th>Evidence</th></tr></thead><tbody id="rows">${rows}</tbody></table></div>
   ${na}<div id="detail" class="small" style="margin-top:10px"></div></div>
  <div class="card"><h2>Missing information</h2><ul>${miss}</ul></div>
  <div class="card"><h2>Patient note</h2><ol id="notelist" class="note">${note}</ol></div>
  <div class="card"><details><summary>Provenance</summary><dl class="kv">${prov}</dl></details></div>`;
  const pick = i => {
    document.querySelectorAll("#rows tr").forEach(t => t.classList.toggle("sel", +t.dataset.i === i));
    const c = r.criteria[i]; highlight(c.evidence_ids);
    $("#detail").innerHTML = `<b>${chip(c.label)}</b> ${esc(c.explanation)}` +
      (Object.keys(c.evidence_text).length ? `<ul>${Object.entries(c.evidence_text).map(([k, v]) => `<li>[${k}] ${esc(v)}</li>`).join("")}</ul>` : "") +
      (c.hallucinated_ids.length ? `<div class="err">The model cited sentence ids that do not exist (${c.hallucinated_ids.join(", ")}); they were dropped.</div>` : "") +
      `<div class="muted">${c.timing.n_new ?? "?"} tokens, ${Math.round((c.timing.prefill_ms || 0) + (c.timing.decode_ms || 0))} ms (${esc(c.mode)})</div>` +
      `<details><summary>Raw model output</summary><pre>${esc(c.raw_output)}</pre></details>`;
    const li = c.evidence_ids.length && document.getElementById("s" + c.evidence_ids[0]); if (li) li.scrollIntoView({ block: "nearest" });
  };
  $("#rows").onclick = e => { const t = e.target.closest("tr"); if (t) pick(+t.dataset.i); };
  $("#rows").onkeydown = e => { if (e.key === "Enter" || e.key === " ") { const t = e.target.closest("tr"); if (t) { e.preventDefault(); pick(+t.dataset.i); } } };
  pick(sel);
}

// lab
function fillLabCrit() {
  const cur = $("#labcrit").value;
  $("#labcrit").innerHTML = crits.map((c, i) => `<option value="${i}">${i + 1}. [${esc(c.type)}] ${esc(c.text.slice(0, 80))}</option>`).join("") || '<option value="">Add criteria on the Screening tab first</option>';
  if (cur) $("#labcrit").value = cur;
}
$("#runlab").onclick = async () => {
  const modes = [...document.querySelectorAll("#labmodes input[type=checkbox]:checked")].map(x => x.value);
  const c = crits[+$("#labcrit").value]; showErr($("#laberr"));
  $("#runlab").disabled = true; $("#labstat").innerHTML = '<span class="spinner"></span> running (the CPU is busy, no_cache is slow)...';
  try {
    const d = await post("/api/lab", { note: $("#note").value, criterion: c, modes, gamma: +$("#labgamma").value || 4 });
    renderLab(d); $("#labstat").textContent = "";
  } catch (e) { showErr($("#laberr"), e); $("#labstat").textContent = ""; }
  $("#runlab").disabled = false;
};
function renderLab(d) {
  const f = (x, n = 1) => x == null ? "n/a" : Number(x).toFixed(n);
  const rows = d.runs.map(r => {
    const t = r.timing, s = t.stats;
    return `<tr><td>${esc(r.mode)}</td><td>${f(t.prefill_ms)}</td><td>${f(t.decode_ms)}</td><td>${f(t.tokens_per_s)}</td><td>${f(t.peak_rss_mb, 0)}</td><td>${f(t.kv_cache_mb, 2)}</td>
    <td>${r.mode === "kv_cached" ? '<span class="chip c-na">reference</span>' : r.equal_to_kv_cached ? '<span class="chip c-ok">tokens identical</span>' : '<span class="chip c-bad">tokens differ</span>'}</td>
    <td>${s && s.acceptance_rate != null ? f(s.acceptance_rate * 100, 1) + "% (gamma " + s.gamma + ")" : "-"}</td></tr>`;
  }).join("");
  $("#labout").innerHTML = `<div style="overflow-x:auto"><table><thead><tr><th>Mode</th><th>Prefill ms</th><th>Decode ms</th><th>tok/s</th><th>Peak RSS MB</th><th>KV MB</th><th>Equality vs kv_cached</th><th>Acceptance</th></tr></thead><tbody>${rows}</tbody></table></div>
  <div class="charts"><div><canvas id="ch1" aria-label="Decode time chart" role="img"></canvas></div><div><canvas id="ch2" aria-label="Tokens per second chart" role="img"></canvas></div></div>
  <p class="small muted">Live single-run measurements on this machine; see stored study results for the repeated, controlled benchmarks.</p>`;
  const labels = d.runs.map(r => r.mode), col = ["#2aa8ff", "#5f7bff", "#7b4dff", "#b07bff"];
  const tick = getComputedStyle(root).getPropertyValue("--muted");
  const mk = (id, label, data) => { if (charts[id]) charts[id].destroy(); charts[id] = new Chart($("#" + id), { type: "bar", data: { labels, datasets: [{ label, data, backgroundColor: col }] },
    options: { plugins: { legend: { labels: { color: tick } } }, scales: { x: { ticks: { color: tick } }, y: { ticks: { color: tick }, beginAtZero: true } } } }); };
  mk("ch1", "Decode time (ms)", d.runs.map(r => r.timing.decode_ms)); mk("ch2", "Tokens per second", d.runs.map(r => r.timing.tokens_per_s));
}
$("#runatt").onclick = async () => {
  const o = $("#attout"); o.hidden = false; o.textContent = "running...";
  try { o.textContent = JSON.stringify(await post("/api/attention", { length: +$("#attlen").value }), null, 2); } catch (e) { o.textContent = e.message; }
};
async function loadStored() {
  try {
    const d = await api("/api/stored");
    $("#stored").innerHTML = Object.entries(d).map(([k, v]) => v.data ? `<details open><summary>${esc(k)} <span class="badge">run ${esc(v.run_id)}</span></summary><pre>${esc(JSON.stringify(v.data, null, 2).slice(0, 3000))}</pre></details>`
      : `<p><b>${esc(k)}</b>: <span class="muted">not available yet</span></p>`).join("");
  } catch (e) { $("#stored").textContent = e.message; }
}

// history
async function loadHistory() {
  try {
    const d = await api("/api/history");
    $("#hist").innerHTML = d.reports.length ? `<table><thead><tr><th>When (UTC)</th><th>Outcome</th><th>Mode</th><th>Criteria</th><th>Note</th><th></th></tr></thead><tbody>` +
      d.reports.map(r => `<tr><td>${esc(r.created)}</td><td>${esc(r.outcome)}</td><td>${esc(r.mode)}</td><td>${r.n_criteria}</td><td>${esc(r.note_preview)}</td>
      <td><button class="btn" data-open="${r.id}">Open</button> <a class="btn" href="/api/report/${r.id}.html" target="_blank" rel="noopener">HTML</a> <a class="btn" href="/api/report/${r.id}.json" download>JSON</a> <button class="btn" data-rm="${r.id}">Delete</button></td></tr>`).join("") + "</tbody></table>"
      : '<p class="muted">No saved reports yet.</p>';
  } catch (e) { $("#hist").innerHTML = `<div class="err">${esc(e.message)}</div>`; }
}
$("#hist").addEventListener("click", async e => {
  const o = e.target.dataset.open, d = e.target.dataset.rm;
  try {
    if (o) { last = await api("/api/history/" + o); document.querySelector('[data-tab=screen]').click(); $("#note").value = last.note_text; crits = last.criteria.map(c => ({ type: c.type, text: c.text })); renderCrits(); countNote(); renderResults(last, 0); }
    if (d) { await api("/api/history/" + d, { method: "DELETE" }); loadHistory(); }
  } catch (er) { $("#hist").insertAdjacentHTML("afterbegin", `<div class="err">${esc(er.message)}</div>`); }
});
