// Decorative 3D scenes (monochrome). Own module so app.js stays readable.
// Rules: 30 fps cap, DPR <= 1.5, paused when tab hidden / off-screen / inference busy, static fallback without WebGL or with reduced motion.
import * as THREE from "./vendor/three.module.min.js";

const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
let webgl = false;
try { const c = document.createElement("canvas"); webgl = !!(c.getContext("webgl2") || c.getContext("webgl")); } catch (e) { }
const scenes = []; let busy = false, last = 0, running = false;
let mx = 0, my = 0;
addEventListener("pointermove", e => { mx = (e.clientX / innerWidth - .5) * 2; my = (e.clientY / innerHeight - .5) * 2; }, { passive: true });
document.addEventListener("kormat:busy", e => { busy = !!e.detail; });
window.kScene = { progress: 0 };

function make(canvas, opts) {
  const r = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: true });
  r.setPixelRatio(Math.min(devicePixelRatio || 1, 1.5));
  const scene = new THREE.Scene(), cam = new THREE.PerspectiveCamera(40, 1, .1, 50); cam.position.z = opts.z;
  const s = { canvas, r, scene, cam, visible: true, enabled: true, update: () => { }, mats: [], color: "" };
  const size = () => { const w = canvas.clientWidth || 300, h = canvas.clientHeight || 200; r.setSize(w, h, false); cam.aspect = w / h; cam.updateProjectionMatrix(); s.dirty = true; };
  size(); new ResizeObserver(size).observe(canvas);
  new IntersectionObserver(es => { s.visible = es[0].isIntersecting; }, {}).observe(canvas);
  scenes.push(s); return s;
}
function recolor(s) {
  const c = css("--text"); if (c === s.color) return; s.color = c;
  s.mats.forEach(m => m.color.set(c)); if (s.scene.fog) s.scene.fog.color.set(css("--surface"));
}
function frame(t) {
  requestAnimationFrame(frame);
  if (document.hidden || busy || t - last < 33) return; last = t;
  scenes.forEach(s => { if (!s.visible || !s.enabled) return; recolor(s); s.update(t / 1000); s.r.render(s.scene, s.cam); });
}
function start() { if (!running && !reduce) { running = true; requestAnimationFrame(frame); } }
function renderOnce(s) { recolor(s); s.update(0); s.r.render(s.scene, s.cam); }

// Loader: double helix of spheres; strand 1 = sentences, strand 2 = criteria, rungs = links.
function loader(canvas) {
  const s = make(canvas, { z: 12 }), N = 22, g = new THREE.Group(); s.scene.add(g);
  const mat = new THREE.MeshBasicMaterial({ transparent: true }), lm = new THREE.LineBasicMaterial({ transparent: true, opacity: .6 });
  s.mats.push(mat, lm);
  const geo = new THREE.SphereGeometry(.2, 12, 8), nodes = [], P = [[], []];
  for (let k = 0; k < 2; k++) for (let i = 0; i < N; i++) {
    const a = i * .5 + k * Math.PI, y = (i - N / 2) * .32, p = new THREE.Vector3(Math.cos(a) * 1.6, y, Math.sin(a) * 1.6);
    P[k].push(p); const m = new THREE.Mesh(geo, mat.clone()); s.mats.push(m.material); m.position.copy(p); g.add(m); nodes.push({ m, i });
  }
  const pts = []; for (let i = 0; i < N; i++) { pts.push(P[0][i], P[1][i]); if (i) pts.push(P[0][i - 1], P[0][i], P[1][i - 1], P[1][i]); }
  g.add(new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(pts), lm));
  g.rotation.z = .25;
  s.update = t => {
    g.rotation.y = t * .5; const p = window.kScene.progress;
    nodes.forEach(({ m, i }) => { const on = i / N <= p; const pulse = on ? 1 + .5 * Math.max(0, Math.sin(t * 3 - i * .5)) : .6; m.scale.setScalar(pulse); m.material.opacity = on ? 1 : .25; });
  };
  s.enabled = true; renderOnce(s); return s;
}
// Hero: wireframe icosahedron + vertex points, gentle mouse parallax (<= 8 degrees), grey fog.
function hero(canvas) {
  const s = make(canvas, { z: 5 }); s.scene.fog = new THREE.Fog(0xffffff, 4, 8);
  const mat = new THREE.MeshBasicMaterial({ wireframe: true, transparent: true, opacity: .8 }), pm = new THREE.PointsMaterial({ size: .06, transparent: true });
  s.mats.push(mat, pm);
  const g = new THREE.Group(), ico = new THREE.Mesh(new THREE.IcosahedronGeometry(1.5, 1), mat), inner = new THREE.Mesh(new THREE.IcosahedronGeometry(.8, 0), mat);
  const pts = new THREE.Points(new THREE.IcosahedronGeometry(2.1, 2), pm); g.add(ico, inner, pts); s.scene.add(g);
  const D = Math.PI / 180; let tx = 0, ty = 0;
  s.update = t => { tx += (my * 8 * D - tx) * .08; ty += (mx * 8 * D - ty) * .08; g.rotation.set(t * .12 + tx, t * .2 + ty, 0); inner.rotation.y = -t * .5; };
  renderOnce(s); return s;
}
const L = document.getElementById("load3d"), H = document.getElementById("hero3d");
if (webgl) {
  try { if (L) loader(L); if (H) hero(H); start(); document.documentElement.classList.add("has3d"); }
  catch (e) { document.documentElement.classList.remove("has3d"); }
}
