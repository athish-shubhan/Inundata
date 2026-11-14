// Inundata 3D viewer. Loads the pipeline export (data/terrain.bin, data/scene.json)
// and draws the study area as a terrain mesh with switchable data layers.
// Every statistic shown is counted in the browser from the exported cells.
import * as THREE from 'three';
import { OrbitControls } from './vendor/OrbitControls.js';

const FLAG_FLOODED = 1, FLAG_WATER = 2;
const COL = {
  hit: [100, 143, 255], miss: [220, 38, 127], falseAlarm: [255, 176, 0],
  water: [28, 58, 110], flood: [64, 156, 255],
};
const $ = (id) => document.getElementById(id);

const state = { mode: 'terrain', handM: 3, threshold: 0, exaggeration: 2.5, touring: false };
let scene3, camera, renderer, controls, terrain, texture, ctx, image;
let S, elev, hand, risk, flags, shade, W, H, N;
let assetMesh, facilityMesh, riverLines, raycaster, pointer = new THREE.Vector2(-9, -9);

init().catch((err) => {
  $('loading').textContent = 'Could not load the viewer: ' + err.message;
  console.error(err);
});

async function init() {
  const [sceneRes, binRes] = await Promise.all([fetch('data/scene.json'), fetch('data/terrain.bin')]);
  if (!sceneRes.ok || !binRes.ok) throw new Error('data files missing; run scripts/export_viewer.py');
  S = await sceneRes.json();
  const buf = await binRes.arrayBuffer();
  W = S.width; H = S.height; N = W * H;
  if (buf.byteLength !== N * 5) throw new Error('terrain.bin does not match scene.json');
  const elevRaw = new Uint16Array(buf, 0, N);
  hand = new Uint8Array(buf, N * 2, N);
  risk = new Uint8Array(buf, N * 3, N);
  flags = new Uint8Array(buf, N * 4, N);
  elev = new Float32Array(N);
  for (let i = 0; i < N; i++) elev[i] = elevRaw[i] / 10; // metres above elev_offset_m
  state.threshold = S.threshold;

  computeHillshade();
  buildScene();
  buildUi();
  paint();
  layoutOverlays();
  $('loading').remove();
  const params = new URLSearchParams(location.search);
  if (params.has('capture')) {
    // Frame-by-frame rendering for scripts/record_viewer.js: the caller supplies
    // the clock, so the recorded tour is smooth however slow the machine is.
    startTour();
    window.inundataFrame = (ms) => { frame(ms); return state.touring; };
    return;
  }
  renderer.setAnimationLoop(frame);
  if (params.has('tour')) startTour();
}

// ---------- terrain colouring ----------

function computeHillshade() {
  shade = new Float32Array(N);
  const cell = S.cell_m, sun = new THREE.Vector3(-0.55, 0.62, -0.56).normalize(); // light from the north-west
  for (let r = 0; r < H; r++) {
    for (let c = 0; c < W; c++) {
      const l = elev[r * W + Math.max(c - 1, 0)], rr = elev[r * W + Math.min(c + 1, W - 1)];
      const u = elev[Math.max(r - 1, 0) * W + c], d = elev[Math.min(r + 1, H - 1) * W + c];
      // surface normal of the height field, z exaggerated so relief reads at 30 m cells
      const nx = (l - rr) * 2.2, nz = (u - d) * 2.2, ny = 2 * cell;
      const inv = 1 / Math.hypot(nx, ny, nz);
      const dot = (nx * sun.x + ny * sun.y + nz * sun.z) * inv;
      shade[r * W + c] = Math.min(1.25, Math.max(0.38, 0.35 + dot * 1.05));
    }
  }
}

const TINT = [[0, 52, 78, 62], [40, 78, 100, 72], [140, 112, 118, 86], [300, 138, 128, 104], [520, 176, 170, 158]];
function tint(h) {
  for (let i = 1; i < TINT.length; i++) {
    if (h <= TINT[i][0]) {
      const a = TINT[i - 1], b = TINT[i], t = (h - a[0]) / (b[0] - a[0]);
      return [a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t, a[3] + (b[3] - a[3]) * t];
    }
  }
  return TINT[TINT.length - 1].slice(1);
}

function riskRamp(t) { // pale yellow -> orange -> deep red
  const stops = [[255, 237, 160], [254, 178, 76], [240, 59, 32], [150, 0, 38]];
  const x = Math.min(0.999, Math.max(0, t)) * 3, i = Math.floor(x), f = x - i;
  return stops[i].map((v, k) => v + (stops[i + 1][k] - v) * f);
}

// Predicted-wet test for the two comparison modes.
function predicted(i) {
  if (state.mode === 'hand') return hand[i] * S.hand_step_m <= state.handM;
  return risk[i] !== S.risk_nodata && risk[i] / 254 >= state.threshold;
}

function paint() {
  const px = image.data, mode = state.mode;
  let hit = 0, miss = 0, fa = 0;
  for (let i = 0; i < N; i++) {
    const base = tint(elev[i]), s = shade[i];
    let r = base[0] * s, g = base[1] * s, b = base[2] * s;
    const blend = (c, a) => { r += (c[0] * s - r) * a; g += (c[1] * s - g) * a; b += (c[2] * s - b) * a; };
    const obs = (flags[i] & FLAG_FLOODED) !== 0;
    if (flags[i] & FLAG_WATER) blend(COL.water, 0.85);

    if (mode === 'observed') {
      if (obs) blend(COL.flood, 0.8);
    } else if (mode === 'risk') {
      if (risk[i] !== S.risk_nodata) {
        const t = risk[i] / 254;
        if (t > 0.04) blend(riskRamp(t), 0.3 + 0.62 * t);
      }
    } else if (mode === 'compare' || mode === 'hand') {
      const valid = mode === 'hand' || risk[i] !== S.risk_nodata;
      if (valid) {
        const pred = predicted(i);
        if (pred && obs) { hit++; blend(COL.hit, 0.85); }
        else if (!pred && obs) { miss++; blend(COL.miss, 0.85); }
        else if (pred && !obs) { fa++; blend(COL.falseAlarm, 0.72); }
      }
    }
    const o = i * 4;
    px[o] = Math.min(255, r); px[o + 1] = Math.min(255, g); px[o + 2] = Math.min(255, b); px[o + 3] = 255;
  }
  ctx.putImageData(image, 0, 0);
  texture.needsUpdate = true;
  updateStats(hit, miss, fa);
  recolourMarkers();
}

// ---------- scene ----------

function buildScene() {
  const host = $('stage');
  renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: false });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  host.appendChild(renderer.domElement);
  scene3 = new THREE.Scene();
  scene3.background = new THREE.Color(0x0b1016);
  scene3.fog = new THREE.Fog(0x0b1016, 22000, 60000);
  camera = new THREE.PerspectiveCamera(38, 1, 50, 120000);

  const cell = S.cell_m, pos = new Float32Array(N * 3), uv = new Float32Array(N * 2);
  for (let r = 0; r < H; r++) {
    for (let c = 0; c < W; c++) {
      const i = r * W + c;
      pos[i * 3] = (c - (W - 1) / 2) * cell;
      pos[i * 3 + 1] = elev[i];
      pos[i * 3 + 2] = (r - (H - 1) / 2) * cell;
      uv[i * 2] = c / (W - 1);
      uv[i * 2 + 1] = 1 - r / (H - 1);
    }
  }
  const idx = new Uint32Array((W - 1) * (H - 1) * 6);
  let k = 0;
  for (let r = 0; r < H - 1; r++) {
    for (let c = 0; c < W - 1; c++) {
      const a = r * W + c, b = a + 1, d = a + W, e = d + 1;
      idx[k++] = a; idx[k++] = d; idx[k++] = b; idx[k++] = b; idx[k++] = d; idx[k++] = e;
    }
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  geo.setAttribute('uv', new THREE.BufferAttribute(uv, 2));
  geo.setIndex(new THREE.BufferAttribute(idx, 1));

  const canvas = document.createElement('canvas');
  canvas.width = W; canvas.height = H;
  ctx = canvas.getContext('2d');
  image = ctx.createImageData(W, H);
  texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.anisotropy = renderer.capabilities.getMaxAnisotropy();
  terrain = new THREE.Mesh(geo, new THREE.MeshBasicMaterial({ map: texture }));
  scene3.add(terrain);

  // A plinth under the terrain so the block reads as a cut-out model.
  const plinth = new THREE.Mesh(
    new THREE.BoxGeometry(W * cell, 400, H * cell),
    new THREE.MeshBasicMaterial({ color: 0x151c25 }));
  plinth.position.y = -200.5;
  scene3.add(plinth);

  const pin = new THREE.ConeGeometry(44, 250, 10); pin.rotateX(Math.PI); pin.translate(0, 125, 0);
  assetMesh = new THREE.InstancedMesh(pin, new THREE.MeshBasicMaterial(), S.assets.length);
  assetMesh.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(S.assets.length * 3), 3);
  facilityMesh = new THREE.InstancedMesh(new THREE.OctahedronGeometry(62), new THREE.MeshBasicMaterial(), S.facilities.length);
  facilityMesh.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(S.facilities.length * 3), 3);
  riverLines = new THREE.LineSegments(new THREE.BufferGeometry(),
    new THREE.LineBasicMaterial({ color: 0x8fd3ff, transparent: true, opacity: 0.9 }));
  scene3.add(assetMesh, facilityMesh, riverLines);

  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.maxPolarAngle = Math.PI * 0.47;
  controls.minDistance = 1500; controls.maxDistance = 40000;
  controls.addEventListener('start', () => { if (state.touring) stopTour(); });
  setView(VIEWS.home);

  raycaster = new THREE.Raycaster();
  renderer.domElement.addEventListener('pointermove', (e) => {
    const rect = renderer.domElement.getBoundingClientRect();
    pointer.set(((e.clientX - rect.left) / rect.width) * 2 - 1, -((e.clientY - rect.top) / rect.height) * 2 + 1);
    pointer.px = e.clientX; pointer.py = e.clientY;
  });
  renderer.domElement.addEventListener('pointerleave', () => pointer.set(-9, -9));
  window.addEventListener('resize', resize);
  resize();
}

function resize() {
  const host = $('stage'), w = host.clientWidth, h = host.clientHeight;
  renderer.setSize(w, h);
  camera.aspect = w / h;
  // Keep the terrain centred in the space the panel leaves free.
  const panelOpen = !document.body.classList.contains('panel-closed') && w > 900;
  if (panelOpen) camera.setViewOffset(w, h, -170, 0, w, h); else camera.clearViewOffset();
  camera.updateProjectionMatrix();
}

// Bilinear ground height (scene units) at a fractional grid position.
function groundY(col, row) {
  const c = Math.min(W - 1.001, Math.max(0, col)), r = Math.min(H - 1.001, Math.max(0, row));
  const c0 = Math.floor(c), r0 = Math.floor(r), fc = c - c0, fr = r - r0, i = r0 * W + c0;
  const top = elev[i] * (1 - fc) + elev[i + 1] * fc, bot = elev[i + W] * (1 - fc) + elev[i + W + 1] * fc;
  return (top * (1 - fr) + bot * fr) * state.exaggeration;
}
const worldX = (col) => (col - (W - 1) / 2) * S.cell_m;
const worldZ = (row) => (row - (H - 1) / 2) * S.cell_m;

// Markers and the river sit on the surface, so they move when the exaggeration changes.
function layoutOverlays() {
  terrain.scale.y = state.exaggeration;
  const m = new THREE.Matrix4();
  S.assets.forEach((a, i) => {
    m.makeTranslation(worldX(a.col), groundY(a.col, a.row), worldZ(a.row));
    assetMesh.setMatrixAt(i, m);
  });
  S.facilities.forEach((f, i) => {
    m.makeTranslation(worldX(f.col), groundY(f.col, f.row) + 190, worldZ(f.row));
    facilityMesh.setMatrixAt(i, m);
  });
  assetMesh.instanceMatrix.needsUpdate = facilityMesh.instanceMatrix.needsUpdate = true;
  assetMesh.computeBoundingSphere(); facilityMesh.computeBoundingSphere();

  const pts = [];
  for (const line of S.river) {
    for (let i = 0; i < line.length - 1; i++) {
      for (const [c, r] of [line[i], line[i + 1]]) pts.push(worldX(c), groundY(c, r) + 12, worldZ(r));
    }
  }
  riverLines.geometry.setAttribute('position', new THREE.Float32BufferAttribute(pts, 3));
}

function recolourMarkers() {
  const c = new THREE.Color();
  S.assets.forEach((a, i) => {
    const flagged = a.risk !== null && a.risk >= state.threshold;
    assetMesh.setColorAt(i, c.set(flagged ? 0xff5a3c : 0xf4f1ea));
  });
  S.facilities.forEach((f, i) => facilityMesh.setColorAt(i, c.set(0xb89bff)));
  assetMesh.instanceColor.needsUpdate = facilityMesh.instanceColor.needsUpdate = true;
}

// ---------- UI ----------

const MODES = {
  terrain: ['Terrain', 'Copernicus 30 m elevation with hillshade. The Kuma River runs east to west through the basin.'],
  observed: ['Observed flood', 'What flooded on 4 July 2020, from the GSI inundation estimate. This is the reference the model is scored against.'],
  risk: ['Model risk', 'Flood susceptibility from the trained model, 0 to 1. Each cell was scored by a model that never saw its 2 km block.'],
  compare: ['Model vs observed', 'Model cells at or above the threshold, compared with what flooded. Drag the threshold to trade misses for false alarms.'],
  hand: ['Terrain baseline vs observed', 'The simplest rule: call a cell flooded if it sits within a set height of the nearest channel (HAND). A terrain index, not a hydraulic simulation.'],
};

function buildUi() {
  $('place').textContent = S.name;
  const modes = $('modes');
  for (const [key, [label]] of Object.entries(MODES)) {
    const b = document.createElement('button');
    b.type = 'button'; b.textContent = label; b.dataset.mode = key;
    b.addEventListener('click', () => { stopTour(); setMode(key); });
    modes.appendChild(b);
  }
  const thr = $('threshold');
  thr.value = state.threshold;
  thr.addEventListener('input', () => { state.threshold = +thr.value; paint(); syncControls(); });
  $('threshold-reset').addEventListener('click', () => { state.threshold = S.threshold; thr.value = S.threshold; paint(); syncControls(); });
  const hs = $('hand');
  hs.value = state.handM;
  hs.addEventListener('input', () => { state.handM = +hs.value; paint(); syncControls(); });
  const ex = $('exaggeration');
  ex.value = state.exaggeration;
  ex.addEventListener('input', () => { state.exaggeration = +ex.value; layoutOverlays(); syncControls(); });
  for (const [id, obj] of [['show-assets', assetMesh], ['show-facilities', facilityMesh], ['show-river', riverLines]]) {
    $(id).addEventListener('change', (e) => { obj.visible = e.target.checked; });
  }
  $('tour').addEventListener('click', () => (state.touring ? stopTour() : startTour()));
  $('home').addEventListener('click', () => { stopTour(); setView(VIEWS.home); });
  $('panel-toggle').addEventListener('click', () => { document.body.classList.toggle('panel-closed'); resize(); });
  setMode(state.mode);
}

function setMode(mode) {
  state.mode = mode;
  for (const b of $('modes').children) b.setAttribute('aria-pressed', String(b.dataset.mode === mode));
  $('mode-note').textContent = MODES[mode][1];
  $('threshold-row').hidden = mode !== 'compare';
  $('hand-row').hidden = mode !== 'hand';
  $('legend-risk').hidden = mode !== 'risk';
  $('legend-observed').hidden = mode !== 'observed';
  $('legend-compare').hidden = !(mode === 'compare' || mode === 'hand');
  paint();
  syncControls();
}

function syncControls() {
  $('threshold-value').textContent = state.threshold.toFixed(2) + (Math.abs(state.threshold - S.threshold) < 0.005 ? ' (operating point)' : '');
  $('hand-value').textContent = state.handM.toFixed(1) + ' m';
  $('exaggeration-value').textContent = state.exaggeration.toFixed(1) + '×';
  $('threshold').value = state.threshold; $('hand').value = state.handM; $('exaggeration').value = state.exaggeration;
}

const km2 = (cells) => (cells * S.cell_m * S.cell_m / 1e6).toFixed(1);
const pct = (a, b) => (b ? Math.round((a / b) * 100) + '%' : 'n/a');

function updateStats(hit, miss, fa) {
  const box = $('stats'), mode = state.mode;
  if (mode === 'compare' || mode === 'hand') {
    const flaggedAssets = S.assets.filter((a) => a.risk !== null && a.risk >= state.threshold).length;
    box.innerHTML =
      `<div class="stat"><b>${pct(hit, hit + miss)}</b><span>of the observed flood is caught (recall)</span></div>` +
      `<div class="stat"><b>${pct(hit, hit + fa)}</b><span>of flagged cells really flooded (precision)</span></div>` +
      `<div class="stat"><b>${km2(hit + fa)} km²</b><span>flagged in total</span></div>` +
      (mode === 'compare' ? `<div class="stat"><b>${flaggedAssets} / ${S.assets.length}</b><span>client assets flagged</span></div>` : '');
  } else if (mode === 'observed') {
    const inFlood = S.assets.filter((a) => a.flooded).length;
    box.innerHTML =
      `<div class="stat"><b>${km2(S.observed_flood_cells)} km²</b><span>observed flooded</span></div>` +
      `<div class="stat"><b>${inFlood} / ${S.assets.length}</b><span>client assets inside the flood</span></div>`;
  } else if (mode === 'risk') {
    const c = S.confusion;
    box.innerHTML =
      `<div class="stat"><b class="cap">${S.model.replace('_', ' ')}</b><span>selected model</span></div>` +
      `<div class="stat"><b>${S.threshold.toFixed(2)}</b><span>operating threshold</span></div>` +
      `<div class="stat"><b>${km2(c.hit + c.false_alarm)} km²</b><span>at or above it</span></div>`;
  } else {
    box.innerHTML =
      `<div class="stat"><b>${km2(N)} km²</b><span>study area, ${S.cell_m} m cells</span></div>` +
      `<div class="stat"><b>${S.elev_min_m}–${S.elev_max_m} m</b><span>elevation range</span></div>`;
  }
}

function updateTooltip() {
  const tip = $('tip');
  if (pointer.x < -1 || state.touring) { tip.hidden = true; return; }
  raycaster.setFromCamera(pointer, camera);
  const hits = raycaster.intersectObjects([assetMesh, facilityMesh].filter((m) => m.visible), false);
  if (!hits.length) { tip.hidden = true; renderer.domElement.style.cursor = ''; return; }
  const h = hits[0];
  let html;
  if (h.object === assetMesh) {
    const a = S.assets[h.instanceId];
    html = `<b>${a.id}</b> · ${a.type}<br>Insured value ¥${(a.value_jpy / 1e6).toFixed(1)}M (simulated)<br>` +
      `Model risk ${a.risk === null ? 'n/a' : a.risk.toFixed(2)}<br>${a.flooded ? 'Inside the observed flood' : 'Outside the observed flood'}`;
  } else {
    const f = S.facilities[h.instanceId];
    html = `<b>${f.name || f.kind}</b><br>${f.kind.replace(/_/g, ' ')}<br>Model risk ${f.risk === null ? 'n/a' : f.risk.toFixed(2)}` +
      `<br>${f.flooded ? 'Inside the observed flood' : 'Outside the observed flood'}`;
  }
  tip.innerHTML = html;
  tip.hidden = false;
  tip.style.left = Math.min(window.innerWidth - 250, pointer.px + 14) + 'px';
  tip.style.top = (pointer.py + 14) + 'px';
  renderer.domElement.style.cursor = 'pointer';
}

// ---------- camera views and the guided tour ----------

const VIEWS = {
  home: { pos: [900, 10200, 17200], target: [0, 0, 700] },
  west: { pos: [-11500, 5200, 7600], target: [-1800, 200, 300] },
  town: { pos: [3900, 3300, 6200], target: [1500, 150, 900] },
  east: { pos: [11800, 5600, 6800], target: [1500, 200, 300] },
  top: { pos: [0, 19500, 4200], target: [0, 0, 300] },
};

function setView(v) {
  camera.position.set(...v.pos);
  controls.target.set(...v.target);
  controls.update();
}

// Each step: seconds, start view, end view, mode, caption, optional animated value.
const TOUR = [
  { t: 5, from: 'home', to: 'west', mode: 'terrain', caption: 'Hitoyoshi basin, Kumamoto. Real 30 m elevation; the Kuma River runs through the valley floor.' },
  { t: 5, from: 'west', to: 'town', mode: 'observed', caption: '4 July 2020: the observed flood extent (GSI estimate). White and red pins are the client’s 120 assets.' },
  { t: 6, from: 'town', to: 'east', mode: 'hand', anim: ['handM', 0, 8], caption: 'Baseline: flag every cell within a few metres of the channel. It catches the flood, with many false alarms (gold).' },
  { t: 5, from: 'east', to: 'top', mode: 'risk', caption: 'The trained model’s flood susceptibility, scored on blocks it never saw in training.' },
  { t: 6, from: 'top', to: 'home', mode: 'compare', caption: 'Model vs observed at the operating threshold: blue caught, magenta missed, gold false alarm.' },
];
let tour = null;

function startTour() {
  state.touring = true;
  tour = { step: -1, started: 0 };
  document.body.classList.add('touring');
  $('tour').textContent = 'Stop tour';
  controls.enabled = false;
}

function stopTour() {
  if (!state.touring) return;
  state.touring = false; tour = null;
  document.body.classList.remove('touring');
  $('tour').textContent = 'Play tour';
  $('caption').hidden = true;
  controls.enabled = true;
  state.threshold = S.threshold; state.handM = 3;
  paint(); syncControls();
  document.body.dataset.tour = 'done';
}

const ease = (x) => x * x * (3 - 2 * x);
const a3 = new THREE.Vector3(), b3 = new THREE.Vector3();

function advanceTour(now) {
  if (tour.step < 0 || (now - tour.started) / 1000 >= TOUR[tour.step].t) {
    tour.step++;
    tour.started = now;
    if (tour.step >= TOUR.length) { stopTour(); setView(VIEWS.home); return; }
    const s = TOUR[tour.step];
    if (s.anim) state[s.anim[0]] = s.anim[1];
    setMode(s.mode);
    $('caption').textContent = s.caption;
    $('caption').hidden = false;
  }
  const s = TOUR[tour.step], x = Math.min(1, (now - tour.started) / 1000 / s.t), e = ease(x);
  const from = VIEWS[s.from], to = VIEWS[s.to];
  camera.position.copy(a3.set(...from.pos).lerp(b3.set(...to.pos), e));
  controls.target.copy(a3.set(...from.target).lerp(b3.set(...to.target), e));
  camera.lookAt(controls.target);
  if (s.anim) {
    const v = s.anim[1] + (s.anim[2] - s.anim[1]) * Math.min(1, x * 1.25);
    if (Math.abs(v - state[s.anim[0]]) > 0.2 || x >= 0.8) { state[s.anim[0]] = Math.round(v * 4) / 4; paint(); syncControls(); }
  }
}

function frame(now) {
  if (state.touring) advanceTour(now); else controls.update();
  updateTooltip();
  renderer.render(scene3, camera);
}
