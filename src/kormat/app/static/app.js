"use strict";
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const ico = n => `<svg class="ic" aria-hidden="true"><use href="#i-${n}"/></svg>`;
const reduce = () => matchMedia("(prefers-reduced-motion: reduce)").matches;
// label -> [css class, icon]; meaning is carried by fill style + icon + text, never colour
const CHIP = { "included": ["c-ok", "check"], "not excluded": ["c-ok", "check"], "not included": ["c-bad", "x"], "excluded": ["c-bad", "x"],
  "not enough information": ["c-warn", "q"], "INVALID": ["c-inv", "warn"], "not applicable": ["c-na", "minus"] };
const chip = (l, lg) => { const [c, i] = CHIP[l] || CHIP["not applicable"]; return `<span class="chip ${c}${lg ? " lg" : ""}">${ico(i)}${esc(l)}</span>`; };
const LEGEND = [["included", "criterion favourable"], ["excluded", "criterion unfavourable"], ["not enough information", "note does not say"], ["not applicable", "ignored"], ["INVALID", "model output unusable"]];
const legendHTML = () => LEGEND.map(([l, d]) => `<span>${chip(l)}<span>${esc(d)}</span></span>`).join("");
let crits = [], last = null, lastLab = null, charts = {}, limits = { max_note_chars: 12000, max_criteria: 40 };

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
const showErr = (el, e) => { el.innerHTML = e ? `<div class="err" role="alert">${ico("warn")}<div><b>Problem:</b> ${esc(e.message)}${e.hint ? `<span class="hint">${esc(e.hint)}</span>` : ""}</div></div>` : ""; };

// legends
$("#legend").innerHTML = legendHTML(); $("#legend2").innerHTML = legendHTML();

// theme
const root = document.documentElement;
const setThemeUI = () => { const d = root.dataset.theme === "dark"; $("#themelbl").textContent = d ? "Light" : "Dark"; $("#theme use").setAttribute("href", d ? "#i-sun" : "#i-moon"); $("#theme").setAttribute("aria-label", d ? "Switch to light theme" : "Switch to dark theme"); };
setThemeUI();
$("#theme").onclick = () => {
  root.dataset.theme = root.dataset.theme === "dark" ? "light" : "dark";
  try { localStorage.setItem("kormat-theme", root.dataset.theme); } catch (e) { }
  setThemeUI(); if (lastLab) drawCharts(lastLab);
};

// tabs
const tabs = $$("nav button");
function openTab(name) {
  tabs.forEach(x => { x.setAttribute("aria-selected", x.dataset.tab === name); x.tabIndex = x.dataset.tab === name ? 0 : -1; });
  ["screen", "lab", "history", "about"].forEach(t => $("#tab-" + t).hidden = t !== name);
  if (name === "history") loadHistory();
  if (name === "lab") { fillLabCrit(); loadStored(); }
}
tabs.forEach(b => { b.onclick = () => openTab(b.dataset.tab); });
$("nav").addEventListener("keydown", e => {
  const i = tabs.indexOf(document.activeElement); if (i < 0 || !["ArrowRight", "ArrowLeft"].includes(e.key)) return;
  const n = tabs[(i + (e.key === "ArrowRight" ? 1 : tabs.length - 1)) % tabs.length]; n.focus(); n.click();
});

// health
async function health() {
  const el = $("#health"), tx = $("#healthtxt");
  try {
    const h = await api("/api/health"); loaderUpdate(h); limits = h.limits; $("#notemax").textContent = limits.max_note_chars;
    const pct = h.progress ? ` ${Math.round(h.progress * 100)}%` : "";
    el.className = "badge " + h.status; tx.textContent = h.status === "ready" ? `Model ready (${h.engine})` : `${h.status}${pct}: ${h.detail}`;
    $("#rulelist").innerHTML = h.rule.split(/\s(?=\d\.\s)/).map(s => `<li><span>${esc(s.replace(/^\d\.\s*/, ""))}</span></li>`).join("");
    if (h.status === "ready") return;
  } catch (e) { el.className = "badge error"; tx.textContent = "Server unreachable"; loaderUpdate(null); }
  setTimeout(health, 1000);
}
health();

// loading overlay: real /api/health state, never blocks forever
const LD = $("#loader"), T0 = performance.now(), SLOW = +(new URLSearchParams(location.search).get("slowload") || 0) * 1000;
let loaderDone = false, creep = 0;
const setScene = p => { window.kScene = window.kScene || {}; window.kScene.progress = p; };
function closeLoader() {
  if (loaderDone) return; loaderDone = true; setScene(1); $("#loadbar").style.width = "100%";
  LD.classList.add("out"); setTimeout(() => { LD.hidden = true; }, reduce() ? 0 : 300);
}
function loaderUpdate(h) {
  if (loaderDone) return; const el = performance.now() - T0;
  creep = Math.min(.9, creep + .02);
  if (!h) { $("#loadtxt").textContent = "Waiting for the server..."; }
  else if (h.status === "ready") {
    $("#loadtxt").textContent = "Ready"; setScene(1); $("#loadbar").style.width = "100%";
    setTimeout(closeLoader, Math.max(300, Math.max(h.engine === "fake" ? 1500 : 400, SLOW) - el)); return;
  } else {
    $("#loadtxt").textContent = h.detail || "Loading..."; const p = Math.max(h.progress || 0, creep);
    setScene(p); $("#loadbar").style.width = Math.round(p * 100) + "%";
  }
  if (el > 60000 && !$("#loadskip").dataset.cont) { $("#loadskip").dataset.cont = 1; $("#loadskip").textContent = "Continue"; $("#loadtxt").textContent = "Still loading \u2014 you can explore the interface meanwhile"; }
}
$("#loadskip").onclick = closeLoader; $("#loadskip").focus();
const busy = b => document.dispatchEvent(new CustomEvent("kormat:busy", { detail: b }));

// samples
let samples = [];
api("/api/samples").then(d => {
  samples = d.samples;
  samples.forEach((s, i) => $("#sample").insertAdjacentHTML("beforeend", `<option value="${i}">${esc(s.id)} (${esc(s.split)})</option>`));
}).catch(() => { });
$("#sample").onchange = e => {
  const s = samples[e.target.value]; if (!s) { $("#samplehelp").textContent = "Fills the note and its criteria below so you can try the tool in one click."; return; }
  $("#note").value = s.note; crits = s.criteria.map(c => ({ ...c })); renderCrits(); countNote();
  $("#samplehelp").textContent = `Loaded ${s.id}: note and ${crits.length} criteria filled in. Edit freely, then screen.`;
};
const countNote = () => { const n = $("#note").value.length; $("#notecount").textContent = n; $("#notecount").style.fontWeight = n > limits.max_note_chars ? "700" : ""; $("#notecount").style.textDecoration = n > limits.max_note_chars ? "underline" : ""; };
$("#note").oninput = countNote;

// decoding mode: radio cards + live consequence text
const MODE_HELP = {
  kv_cached: "You chose Fast: each criterion is answered with cached keys/values, the default and the quickest exact path.",
  no_cache: "You chose Reference: expect roughly 10-30x longer per criterion, in exchange for a from-scratch computation that proves the fast path exact.",
  speculative: "You chose Speculative: the 0.5B draft model proposes γ tokens per round and the 1.5B model checks them. Output is identical; on this CPU it only helps long answers.",
  hf_generate: "You chose Library baseline: Hugging Face generate() decodes, useful only to compare against our own implementations."
};
const getMode = () => ($("input[name=mode]:checked") || {}).value || "kv_cached";
function updateMode() {
  const m = getMode(); $("#modehelp").textContent = MODE_HELP[m]; $("#gammabox").hidden = m !== "speculative";
}
$("#modes").addEventListener("change", updateMode); updateMode();

// criteria rows
function updateRunLabel() {
  const n = crits.length; $("#critcount").textContent = n;
  $("#runlbl").textContent = `Screen patient against ${n} criteri${n === 1 ? "on" : "a"}`;
}
function renderCrits() {
  $("#crits").innerHTML = crits.map((c, i) => `<div class="crit${c.type === "exclusion" ? " excl" : ""}"><select aria-label="Criterion ${i + 1} type" data-i="${i}" data-f="type">
   <option value="inclusion" ${c.type === "inclusion" ? "selected" : ""}>Inclusion — patient must meet this</option><option value="exclusion" ${c.type === "exclusion" ? "selected" : ""}>Exclusion — patient must NOT have this</option></select>
   <input aria-label="Criterion ${i + 1} text" data-i="${i}" data-f="text" value="${esc(c.text)}"><button class="btn" type="button" data-del="${i}" aria-label="Remove criterion ${i + 1}">${ico("x")}</button></div>`).join("")
    || '<p class="help">No criteria yet. Split pasted text, fetch an NCT trial, or add one by hand.</p>';
  updateRunLabel();
}
$("#crits").addEventListener("input", e => { const t = e.target; if (t.dataset.f) { crits[t.dataset.i][t.dataset.f] = t.value; if (t.dataset.f === "type") t.closest(".crit").classList.toggle("excl", t.value === "exclusion"); } });
$("#crits").addEventListener("click", e => { const b = e.target.closest("[data-del]"); if (b) { crits.splice(+b.dataset.del, 1); renderCrits(); } });
$("#addrow").onclick = () => { crits.push({ type: "inclusion", text: "" }); renderCrits(); const ins = $$("#crits input[data-f=text]"); ins[ins.length - 1].focus(); };
$("#clearrows").onclick = () => { if (crits.length && !confirm(`Remove all ${crits.length} criteria? This cannot be undone.`)) return; crits = []; renderCrits(); };
$("#parse").onclick = async () => {
  try { showErr($("#screenerr")); const d = await post("/api/parse_criteria", { block: $("#block").value }); crits = crits.concat(d.criteria); renderCrits(); }
  catch (e) { showErr($("#screenerr"), e); }
};
$("#fetchnct").onclick = async () => {
  const b = $("#fetchnct"); b.disabled = true;
  try { showErr($("#screenerr")); const d = await api("/api/nct/" + encodeURIComponent($("#nct").value.trim())); $("#block").value = d.block; crits = d.criteria; renderCrits(); }
  catch (e) { showErr($("#screenerr"), e); }
  b.disabled = false;
};
renderCrits();

// screening, with a determinate per-criterion progress estimate (server answers once at the end)
let perCrit = 4000;
$("#run").onclick = async () => {
  const b = $("#run"), n = crits.length, p = $("#prog"), bar = $("#progbar"); b.disabled = true; showErr($("#screenerr"));
  const oldLbl = $("#runlbl").textContent; $("#runlbl").textContent = "Screening..."; b.querySelector("svg").outerHTML = '<span class="spinner"></span>';
  p.hidden = false; bar.style.width = "0%"; $("#results").innerHTML = '<div class="card"><h2>Screening...</h2><div class="sk"></div><div class="sk" style="width:80%"></div><div class="sk" style="width:90%"></div><div class="sk" style="width:60%"></div></div>';
  const t0 = performance.now(), tick = () => {
    const k = Math.min(n - 1, Math.floor((performance.now() - t0) / perCrit)) + 1;
    $("#runstat").textContent = `Screening criterion ${Math.min(k, n)} of ${n}...`; bar.style.width = `${Math.round((k - 1) / Math.max(n, 1) * 100)}%`;
  };
  tick(); const iv = setInterval(tick, 300); busy(true);
  try {
    last = await post("/api/screen", { note: $("#note").value, criteria: crits, mode: getMode(), gamma: +$("#gamma").value || 4 });
    clearInterval(iv); bar.style.width = "100%"; if (n) perCrit = Math.max(500, last.provenance.total_ms / n);
    renderResults(last, 0);
    $("#runstat").textContent = `Done: ${n} of ${n} criteria in ${(last.provenance.total_ms / 1000).toFixed(1)} s`;
  } catch (e) {
    clearInterval(iv); showErr($("#screenerr"), e); p.hidden = true;
    $("#results").innerHTML = '<div class="card empty"><h2>No result</h2><p class="help">The screening did not finish. Fix the problem shown next to the button and try again.</p></div>';
    if (e.status === 503) setTimeout(health, 100);
  }
  busy(false); b.querySelector(".spinner").outerHTML = ico("play"); updateRunLabel(); b.disabled = false;
};
function highlight(ids) {
  $$("#notelist li").forEach(li => li.classList.toggle("hl", ids.includes(+li.dataset.id)));
}
function countUp(el, to) {
  if (reduce() || !to) { el.textContent = to; return; }
  const t0 = performance.now(); (function f(t) { const k = Math.min(1, (t - t0) / 600); el.textContent = Math.round(to * k); if (k < 1) requestAnimationFrame(f); })(t0);
}
function renderResults(r, sel) {
  const a = r.aggregate, lab = a.outcome.startsWith("Likely") ? "excluded" : a.outcome.startsWith("Undet") ? "not enough information" : "included";
  const rows = r.criteria.map((c, i) => `<tr tabindex="0" data-i="${i}" style="--i:${i}" class="${i === sel ? "sel" : ""}" aria-label="Criterion ${i + 1}, ${esc(c.label)}">
   <td>${i + 1}</td><td>${esc(c.type)}</td><td>${esc(c.text)}</td><td>${chip(c.label)}</td><td>${c.evidence_ids.map(x => "[" + x + "]").join(" ") || '<span class="help">none</span>'}</td></tr>`).join("");
  const miss = a.missing_information.map(m => `<li>${esc(m.text)} <span class="help">(${esc(m.type)})</span></li>`).join("") || "<li class='help'>None</li>";
  const na = a.ignored_not_applicable.length ? `<p class="help">Not applicable (ignored in the decision): ${a.ignored_not_applicable.map(m => esc(m.text)).join("; ")}</p>` : "";
  const note = Object.entries(r.sentences).map(([k, v]) => `<li data-id="${k}" id="s${k}"><span class="n">${k}</span>${esc(v)}</li>`).join("");
  const prov = Object.entries(r.provenance).map(([k, v]) => `<dt>${esc(k)}</dt><dd>${esc(v ?? "n/a")}</dd>`).join("");
  const nIncl = r.criteria.length;
  $("#results").innerHTML = `
  <div class="card outcome"><p class="help">Preliminary outcome (screening aid, not ground truth) from <b class="count" id="cu">${nIncl}</b> criteria</p>
   <div class="big">${chip(lab, true)} <b style="font-size:24px;margin-left:8px">${esc(a.outcome)}</b></div><p class="why">${esc(a.reason)}</p>
   <details><summary>How this was decided</summary><p class="help">${esc(a.rule)}</p></details>
   <div class="row no-print"><a class="btn" href="/api/report/${r.id}.html" target="_blank" rel="noopener">${ico("file")}Open printable report (save as PDF)</a>
   <a class="btn" href="/api/report/${r.id}.json" download>${ico("down")}Download machine-readable JSON</a></div></div>
  <div class="card"><h2>Criterion results</h2><p class="help" style="margin:-8px 0 8px">Select a row (click, Enter or Space) to highlight the note sentences it cites.</p>
   <div class="tw"><table><thead><tr><th>#</th><th>Type</th><th>Criterion</th><th>Label</th><th>Evidence</th></tr></thead><tbody id="rows">${rows}</tbody></table></div>
   ${na}<div id="detail" style="margin-top:16px"></div></div>
  <div class="card"><h2>Missing information</h2><ul>${miss}</ul></div>
  <div class="card"><h2>Patient note</h2><ol id="notelist" class="note">${note}</ol></div>
  <div class="card"><details><summary>Provenance</summary><dl class="kv">${prov}</dl></details></div>`;
  countUp($("#cu"), nIncl);
  const pick = (i, scroll) => {
    $$("#rows tr").forEach(t => t.classList.toggle("sel", +t.dataset.i === i));
    const c = r.criteria[i]; highlight(c.evidence_ids);
    $("#detail").innerHTML = `<div>${chip(c.label)}</div><p>${esc(c.explanation)}</p>` +
      (Object.keys(c.evidence_text).length ? `<ul>${Object.entries(c.evidence_text).map(([k, v]) => `<li>[${k}] ${esc(v)}</li>`).join("")}</ul>` : "") +
      (c.hallucinated_ids.length ? `<div class="err">${ico("warn")}<div>The model cited sentence ids that do not exist (${c.hallucinated_ids.join(", ")}); they were dropped.</div></div>` : "") +
      `<p class="help">${c.timing.n_new ?? "?"} tokens, ${Math.round((c.timing.prefill_ms || 0) + (c.timing.decode_ms || 0))} ms (${esc(c.mode)})</p>` +
      `<details><summary>Raw model output</summary><pre>${esc(c.raw_output)}</pre></details>`;
    const li = c.evidence_ids.length && document.getElementById("s" + c.evidence_ids[0]);
    if (li && scroll) { const l = $("#notelist"); l.scrollTo({ top: Math.max(0, li.offsetTop - l.offsetTop - 40), behavior: reduce() ? "auto" : "smooth" }); }
  };
  $("#rows").onclick = e => { const t = e.target.closest("tr"); if (t) pick(+t.dataset.i, true); };
  $("#rows").onkeydown = e => { if (e.key === "Enter" || e.key === " ") { const t = e.target.closest("tr"); if (t) { e.preventDefault(); pick(+t.dataset.i, true); } } };
  pick(sel, false); requestAnimationFrame(() => requestAnimationFrame(() => pick(sel, true)));
}

// lab
function fillLabCrit() {
  const cur = $("#labcrit").value;
  $("#labcrit").innerHTML = crits.map((c, i) => `<option value="${i}">${i + 1}. [${esc(c.type)}] ${esc(c.text.slice(0, 80))}</option>`).join("") || '<option value="">Add criteria on the Screen a patient tab first</option>';
  if (cur) $("#labcrit").value = cur;
}
$("#runlab").onclick = async () => {
  const modes = $$("#labmodes input[type=checkbox]:checked").map(x => x.value);
  const c = crits[+$("#labcrit").value]; showErr($("#laberr"));
  $("#runlab").disabled = true; $("#labstat").innerHTML = '<span class="spinner" style="display:inline-block;vertical-align:-3px"></span> Running... the CPU is busy; Reference (no cache) is slow.';
  $("#labout").innerHTML = '<div class="sk"></div><div class="sk" style="width:70%"></div>'; busy(true);
  try {
    const d = await post("/api/lab", { note: $("#note").value, criterion: c, modes, gamma: +$("#labgamma").value || 4, max_new_tokens: +$("#labmax").value || 64 });
    renderLab(d); $("#labstat").textContent = "";
  } catch (e) { showErr($("#laberr"), e); $("#labout").innerHTML = ""; $("#labstat").textContent = ""; }
  busy(false); $("#runlab").disabled = false;
};
const MODE_NAME = { kv_cached: "Fast (KV)", no_cache: "Reference", speculative: "Speculative", hf_generate: "Library" };
function renderLab(d) {
  lastLab = d;
  const f = (x, n = 1) => x == null ? "n/a" : Number(x).toFixed(n);
  const rows = d.runs.map(r => {
    const t = r.timing, s = t.stats;
    return `<tr><td>${esc(MODE_NAME[r.mode] || r.mode)}</td><td>${f(t.prefill_ms)}</td><td>${f(t.decode_ms)}</td><td>${f(t.tokens_per_s)}</td><td>${f(t.peak_rss_mb, 0)}</td><td>${f(t.kv_cache_mb, 2)}</td>
    <td>${r.mode === "kv_cached" ? '<span class="chip c-na">' + ico("minus") + 'reference</span>' : r.equal_to_kv_cached ? '<span class="chip c-ok">' + ico("check") + 'tokens identical</span>' : '<span class="chip c-bad">' + ico("x") + 'tokens differ</span>'}</td>
    <td>${s && s.acceptance_rate != null ? f(s.acceptance_rate * 100, 1) + "% (γ " + s.gamma + ")" : "-"}</td></tr>`;
  }).join("");
  $("#labout").innerHTML = `<div class="tw"><table><thead><tr><th>Mode</th><th>Prefill ms</th><th>Decode ms</th><th>tok/s</th><th>Peak RSS MB</th><th>KV MB</th><th>Equality vs Fast</th><th>Acceptance</th></tr></thead><tbody>${rows}</tbody></table></div>
  <div class="charts"><div><canvas id="ch1" aria-label="Decode time chart" role="img"></canvas></div><div><canvas id="ch2" aria-label="Tokens per second chart" role="img"></canvas></div></div>
  <p class="help">Live single-run measurements on this machine; see stored study results for the repeated, controlled benchmarks.</p>`;
  drawCharts(d);
}
// greys only; bars differ by fill pattern and are labelled directly
function drawCharts(d) {
  if (!$("#ch1")) return;
  const css = n => getComputedStyle(root).getPropertyValue(n).trim();
  const ink = css("--text"), bg = css("--bg"), mid = css("--border-strong"), grid = css("--border"), tx = css("--text-2");
  const pat = k => {
    const c = document.createElement("canvas"); c.width = c.height = 10; const x = c.getContext("2d");
    x.fillStyle = [ink, bg, mid, bg][k % 4]; x.fillRect(0, 0, 10, 10); x.strokeStyle = [bg, ink, ink, ink][k % 4]; x.lineWidth = 1.5;
    if (k % 4 === 1) { x.beginPath(); x.moveTo(0, 10); x.lineTo(10, 0); x.stroke(); }
    if (k % 4 === 3) { x.beginPath(); x.moveTo(5, 0); x.lineTo(5, 10); x.moveTo(0, 5); x.lineTo(10, 5); x.stroke(); }
    return c.getContext("2d").canvas;
  };
  const labels = d.runs.map(r => MODE_NAME[r.mode] || r.mode);
  const valLabels = { id: "vl", afterDatasetsDraw(ch) { const { ctx } = ch; ctx.save(); ctx.fillStyle = ink; ctx.font = "600 12px Inter,system-ui,sans-serif"; ctx.textAlign = "center";
    ch.getDatasetMeta(0).data.forEach((b, i) => { const v = ch.data.datasets[0].data[i]; if (v != null) ctx.fillText(Number(v).toFixed(1), b.x, b.y - 6); }); ctx.restore(); } };
  const mk = (id, label, data) => {
    if (charts[id]) charts[id].destroy();
    const cx = $("#" + id).getContext("2d");
    charts[id] = new Chart($("#" + id), { type: "bar", plugins: [valLabels], data: { labels, datasets: [{ label, data, backgroundColor: data.map((_, i) => cx.createPattern(pat(i), "repeat")), borderColor: ink, borderWidth: 2 }] },
      options: { animation: reduce() ? false : { duration: 700, easing: "easeOutCubic" }, layout: { padding: { top: 20 } },
        plugins: { legend: { labels: { color: tx, font: { family: "Inter, system-ui" } } } },
        scales: { x: { ticks: { color: tx }, grid: { display: false }, border: { color: mid } }, y: { ticks: { color: tx }, grid: { color: grid }, border: { display: false }, beginAtZero: true } } } });
  };
  mk("ch1", "Decode time (ms)", d.runs.map(r => r.timing.decode_ms)); mk("ch2", "Tokens per second", d.runs.map(r => r.timing.tokens_per_s));
}
$("#runatt").onclick = async () => {
  const o = $("#attout"); o.hidden = false; o.textContent = "Running...";
  try { o.textContent = JSON.stringify(await post("/api/attention", { length: +$("#attlen").value }), null, 2); } catch (e) { o.textContent = e.message; }
};
async function loadStored() {
  try {
    const d = await api("/api/stored");
    $("#stored").innerHTML = Object.entries(d).map(([k, v]) => v.data ? `<details><summary>${esc(k)}  <span class="badge">run ${esc(v.run_id)}</span></summary><pre>${esc(JSON.stringify(v.data, null, 2).slice(0, 3000))}</pre></details>`
      : `<p><b>${esc(k)}</b>: <span class="help">not available yet</span></p>`).join("");
  } catch (e) { $("#stored").textContent = e.message; }
}

// history
async function loadHistory() {
  try {
    const d = await api("/api/history");
    $("#hist").innerHTML = d.reports.length ? `<div class="tw"><table><thead><tr><th>When (UTC)</th><th>Outcome</th><th>Mode</th><th>Criteria</th><th>Note</th><th></th></tr></thead><tbody>` +
      d.reports.map(r => `<tr><td>${esc(r.created)}</td><td>${esc(r.outcome)}</td><td>${esc(r.mode)}</td><td>${r.n_criteria}</td><td>${esc(r.note_preview)}</td>
      <td><div class="hist-actions"><button class="btn" type="button" data-open="${r.id}">Open</button><a class="btn" href="/api/report/${r.id}.html" target="_blank" rel="noopener">HTML</a><a class="btn" href="/api/report/${r.id}.json" download>JSON</a><button class="btn" type="button" data-rm="${r.id}">Delete</button></div></td></tr>`).join("") + "</tbody></table></div>"
      : '<p class="help">No saved reports yet. Screen a patient and the report is saved here.</p>';
  } catch (e) { showErr($("#hist"), e); }
}
$("#hist").addEventListener("click", async e => {
  const b = e.target.closest("button"); if (!b) return; const o = b.dataset.open, d = b.dataset.rm;
  try {
    if (o) { last = await api("/api/history/" + o); openTab("screen"); $("#note").value = last.note_text; crits = last.criteria.map(c => ({ type: c.type, text: c.text })); renderCrits(); countNote(); renderResults(last, 0); }
    if (d) { if (!confirm("Delete this saved report?")) return; await api("/api/history/" + d, { method: "DELETE" }); loadHistory(); }
  } catch (er) { $("#hist").insertAdjacentHTML("afterbegin", `<div class="err" role="alert">${ico("warn")}<div>${esc(er.message)}</div></div>`); }
});
