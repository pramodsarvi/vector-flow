/**
 * Application shell: scan library, data loading, and wiring between the 2D
 * plan, the 3D scene and the inspector.
 *
 * All scan data is fetched from the API by scan id. The viewer keeps no copies
 * of floorplan.json next to itself, which is what previously let the page show
 * one scan's geometry while the engine had already produced another.
 */

import { api, ApiError, token } from './api.js';
import { $, clear, el, emptyPanel, errorPanel, fmt, spinner, toast } from './ui.js';
import { Plan2D } from './plan2d.js';
import { MODE, Scene3D } from './scene3d.js';

const POLL_MS = 1500;

const state = {
  config: null,
  scans: [],
  scanId: null,
  scan: null,
  floorplan: null,
  colliders: null,
  selection: null,
  pollTimer: null,
  loadToken: 0,
};

let plan = null;
let scene = null;

/* ------------------------------------------------------------------- boot */

async function boot() {
  wireStaticControls();
  try {
    state.config = await api.config();
  } catch (err) {
    showFatal(err.message, boot);
    return;
  }

  if (state.config.auth_required && !token.get()) {
    if (!(await promptForToken())) return;
  }

  try {
    await refreshScans({ selectFirst: true });
  } catch (err) {
    if (err instanceof ApiError && err.status === 401) {
      token.set('');
      if (await promptForToken()) return refreshScans({ selectFirst: true });
      return;
    }
    showFatal(err.message, () => refreshScans({ selectFirst: true }));
  }
}

async function promptForToken() {
  const supplied = window.prompt('This server requires an API token (SCANNER_TOKEN):');
  if (!supplied) {
    showFatal('An API token is required to use this viewer.', boot);
    return false;
  }
  token.set(supplied.trim());
  try {
    await api.health();
    return true;
  } catch {
    token.set('');
    showFatal('That token was rejected.', boot);
    return false;
  }
}

function showFatal(message, retry) {
  const host = $('#library-list');
  if (host) clear(host).append(errorPanel(message, retry));
  setConnection('error', 'offline');
  toast(message, 'error', 8000);
}

/* ------------------------------------------------------------- scan list */

async function refreshScans({ selectFirst = false } = {}) {
  const { scans } = await api.listScans();
  state.scans = scans;
  renderLibrary();
  setConnection('ok', `${fmt.count(scans.length, 'scan')}`);

  if (selectFirst && !state.scanId) {
    const ready = scans.find((s) => s.status === 'ready') || scans[0];
    if (ready) await selectScan(ready.id);
    else renderEmptyWorkspace();
  }
  // Keep polling while anything is in flight.
  const busy = scans.some((s) => s.status === 'queued' || s.status === 'processing');
  schedulePoll(busy);
}

function schedulePoll(busy) {
  clearTimeout(state.pollTimer);
  if (!busy) return;
  state.pollTimer = setTimeout(async () => {
    try {
      const previous = new Map(state.scans.map((s) => [s.id, s.status]));
      await refreshScans();
      // When the selected scan just became ready, load its geometry.
      for (const scan of state.scans) {
        const was = previous.get(scan.id);
        if (was !== scan.status && scan.status === 'ready') {
          toast(`“${scan.name}” finished processing.`, 'success');
          if (!state.scanId || state.scanId === scan.id) await selectScan(scan.id);
        } else if (was !== scan.status && scan.status === 'failed') {
          toast(`“${scan.name}” failed: ${scan.error || 'unknown error'}`, 'error', 9000);
        }
      }
      if (state.scanId) {
        const current = state.scans.find((s) => s.id === state.scanId);
        if (current) renderScanHeader(current);
      }
    } catch {
      /* transient: the next tick retries */
    }
  }, POLL_MS);
}

function renderLibrary() {
  const host = clear($('#library-list'));
  if (!state.scans.length) {
    host.append(emptyPanel('No scans yet.'));
    return;
  }
  for (const scan of state.scans) {
    host.append(libraryRow(scan));
  }
}

function libraryRow(scan) {
  const active = scan.id === state.scanId;
  const busy = scan.status === 'queued' || scan.status === 'processing';
  const metrics = scan.summary;

  const row = el('div.scan-row', {
    class: `scan-row${active ? ' is-active' : ''}`,
    role: 'option',
    'aria-selected': active ? 'true' : 'false',
    tabindex: '0',
    onclick: () => selectScan(scan.id),
    onkeydown: (event) => {
      if (event.key === 'Enter' || event.key === ' ') {
        event.preventDefault();
        selectScan(scan.id);
      }
    },
  });

  row.append(
    el('div.scan-row-head', {},
      el('span.scan-name', { title: scan.name }, scan.name),
      el(`span.status.status-${scan.status}`, {}, scan.status),
    ),
    el('div.scan-row-meta', {},
      metrics?.area_sqm ? el('span', {}, `${metrics.area_sqm} m²`) : null,
      el('span', {}, fmt.when(scan.created_at)),
      scan.source === 'sample' ? el('span.tag', {}, 'sample') : null,
    ),
  );

  if (busy) {
    const pct = Math.round((scan.progress.step / scan.progress.total) * 100);
    row.append(
      el('div.progress', { role: 'progressbar', 'aria-valuenow': pct, 'aria-valuemin': 0, 'aria-valuemax': 100 },
        el('div.progress-bar', { style: `width:${Math.max(pct, 6)}%` }),
      ),
      el('p.scan-row-note', {}, scan.progress.message),
    );
  } else if (scan.status === 'failed') {
    row.append(el('p.scan-row-note.is-error', {}, scan.error || 'Processing failed'));
  }

  row.append(
    el('div.scan-row-actions', {},
      el('button.btn.tiny', {
        type: 'button',
        disabled: busy,
        title: 'Run the pipeline again on this scan',
        onclick: (event) => {
          event.stopPropagation();
          reprocess(scan);
        },
      }, 'Reprocess'),
      el('button.btn.tiny.danger', {
        type: 'button',
        title: 'Delete this scan and its artifacts',
        onclick: (event) => {
          event.stopPropagation();
          removeScan(scan);
        },
      }, 'Delete'),
    ),
  );
  return row;
}

async function reprocess(scan) {
  try {
    await api.reprocessScan(scan.id);
    toast(`Re-processing “${scan.name}”…`, 'info');
    await refreshScans();
  } catch (err) {
    toast(err.message, 'error', 8000);
  }
}

async function removeScan(scan) {
  if (!window.confirm(`Delete “${scan.name}” and all of its artifacts? This cannot be undone.`)) return;
  try {
    await api.deleteScan(scan.id);
    toast(`Deleted “${scan.name}”.`, 'success');
    if (state.scanId === scan.id) {
      state.scanId = null;
      state.scan = null;
      renderEmptyWorkspace();
    }
    await refreshScans({ selectFirst: true });
  } catch (err) {
    toast(err.message, 'error', 8000);
  }
}

/* ------------------------------------------------------- scan selection */

async function selectScan(scanId) {
  state.scanId = scanId;
  renderLibrary();

  const scan = state.scans.find((s) => s.id === scanId);
  state.scan = scan || null;
  if (scan) renderScanHeader(scan);

  if (!scan || scan.status !== 'ready') {
    renderPending(scan);
    return;
  }

  // Guard against an out-of-order response when the user clicks quickly.
  const loadId = ++state.loadToken;
  showWorkspaceLoading();

  try {
    const [floorplan, colliders] = await Promise.all([
      api.floorplan(scanId),
      api.colliders(scanId).catch(() => ({ colliders: [] })), // optional
    ]);
    if (loadId !== state.loadToken) return;
    state.floorplan = floorplan;
    state.colliders = colliders;
    renderWorkspace();
  } catch (err) {
    if (loadId !== state.loadToken) return;
    clear($('#plan-host')).append(errorPanel(err.message, () => selectScan(scanId)));
  }
}

function renderScanHeader(scan) {
  $('#scan-title').textContent = scan.name;
  $('#scan-sub').textContent =
    `${scan.status === 'ready' ? fmt.duration(scan.duration_s) : scan.status} · ${fmt.when(scan.created_at)}`;

  const summary = scan.summary || {};
  setBadge('badge-area', fmt.area(summary.area_sqm, summary.area_sqft));
  setBadge('badge-perimeter', summary.perimeter_m ? `${summary.perimeter_m} m` : '—');
  setBadge('badge-openings', `${summary.door_count ?? 0} / ${summary.window_count ?? 0}`);
  setBadge('badge-furniture', String(summary.furniture_count ?? 0));
  setBadge('badge-points', fmt.points(summary.point_count_fused));

  renderExports(scan);
}

function setBadge(id, value) {
  const node = $(`#${id}`);
  if (node) node.textContent = value;
}

function renderExports(scan) {
  const host = clear($('#export-menu'));
  const files = [
    ['floorplan.pdf', 'Blueprint PDF'],
    ['floorplan.dxf', 'AutoCAD DXF'],
    ['floorplan.svg', 'Vector SVG'],
    ['floorplan.json', 'CAD JSON'],
    ['room_model.glb', '3D model (GLB)'],
    ['room_model.obj', '3D model (OBJ)'],
    ['aligned_point_cloud.ply', 'Point cloud (PLY)'],
    ['colliders.json', 'Colliders JSON'],
  ];
  const available = new Set(scan.artifacts || []);
  for (const [name, label] of files) {
    if (!available.has(name)) continue;
    host.append(el('a.menu-item', {
      href: api.artifactUrl(scan.id, name),
      download: `${scan.id}-${name}`,
    }, label));
  }
  if (!host.childElementCount) host.append(el('p.menu-empty', {}, 'No artifacts yet'));
}

/* ----------------------------------------------------------- workspace */

function renderEmptyWorkspace() {
  teardownViews();
  $('#scan-title').textContent = 'No scan selected';
  $('#scan-sub').textContent = 'Create one to get started';
  for (const id of ['badge-area', 'badge-perimeter', 'badge-openings', 'badge-furniture', 'badge-points']) {
    setBadge(id, '—');
  }
  clear($('#export-menu')).append(el('p.menu-empty', {}, 'No artifacts yet'));
  clear($('#plan-host')).append(emptyPanel(
    'Process a scan to see its floor design.',
    el('button.btn.primary', { type: 'button', onclick: createSample }, 'Process the sample scan'),
  ));
  clear($('#inspector-body')).append(emptyPanel('Nothing selected.'));
}

function renderPending(scan) {
  teardownViews();
  const host = clear($('#plan-host'));
  if (!scan) {
    host.append(emptyPanel('This scan no longer exists.'));
    return;
  }
  if (scan.status === 'failed') {
    host.append(errorPanel(
      `Processing failed — ${scan.error || 'unknown error'}`,
      () => reprocess(scan),
    ));
    return;
  }
  host.append(spinner(`${scan.progress.message}… (step ${scan.progress.step} of ${scan.progress.total})`));
}

function showWorkspaceLoading() {
  teardownViews();
  clear($('#plan-host')).append(spinner('Loading floor design'));
}

function teardownViews() {
  plan?.destroy();
  plan = null;
  scene?.destroy();
  scene = null;
  // Remove only the canvas: #canvas-host also holds the mode hint elements,
  // and clearing the whole host would delete those out from under the scene.
  $('#canvas3d')?.remove();
  if ($('#measure-list')) clear($('#measure-list'));
  for (const id of ['#measure-hint', '#walk-hint']) {
    const node = $(id);
    if (node) node.hidden = true;
  }
}

function renderWorkspace() {
  teardownViews();

  const planHost = clear($('#plan-host'));
  plan = new Plan2D(planHost, { onSelect: showSelection });
  plan.render(state.floorplan);
  syncLayerToggles();

  const canvas = el('canvas', { id: 'canvas3d', 'aria-label': 'Three-dimensional room view' });
  $('#canvas-host').prepend(canvas);
  scene = new Scene3D(canvas, {
    onMeasure: renderMeasurements,
    onPinClick: showFrame,
    onModeChange: (mode) => {
      for (const button of document.querySelectorAll('[data-mode]')) {
        const isActive = button.dataset.mode === mode;
        button.classList.toggle('is-active', isActive);
        button.setAttribute('aria-pressed', isActive ? 'true' : 'false');
      }
      const measureHint = $('#measure-hint');
      const walkHint = $('#walk-hint');
      if (measureHint) measureHint.hidden = mode !== MODE.MEASURE;
      if (walkHint) walkHint.hidden = mode !== MODE.WALK;
    },
  });
  scene.load(state.floorplan, state.colliders);
  renderMeasurements([], { pending: false });
  showSelection(null);
}

/* ---------------------------------------------------------- inspector */

function showSelection(selection) {
  state.selection = selection;
  const body = clear($('#inspector-body'));
  if (!selection) {
    body.append(emptyPanel('Select a wall, opening or furniture item in the plan.'));
    return;
  }
  const { kind, data } = selection;
  const rows = [];

  if (kind === 'wall') {
    rows.push(['Element', 'Wall'], ['Id', data.wall_id],
      ['Length', `${data.length_m} m (${data.length_ft} ft)`],
      ['Start', `x ${data.start.x}, z ${data.start.z}`],
      ['End', `x ${data.end.x}, z ${data.end.z}`]);
  } else if (kind === 'opening') {
    rows.push(['Element', data.type === 'door' ? 'Door' : 'Window'],
      ['On wall', data.wall],
      ['Width', `${data.width_m} m`],
      ['Height', `${data.height_m} m`],
      ['Sill height', `${data.sill_height_m} m`]);
    if (data.swing) rows.push(['Swing', String(data.swing).replace(/_/g, ' ')]);
  } else if (kind === 'furniture') {
    const b = data.bbox;
    rows.push(['Element', data.type],
      ['Footprint', `${b.width_m} × ${b.depth_m} m`],
      ['Height', `${b.height_m} m`],
      ['Centre', `x ${b.center_x}, z ${b.center_z}`],
      ['Area', `${(b.width_m * b.depth_m).toFixed(2)} m²`]);
  }

  const table = el('dl.kv');
  for (const [label, value] of rows) {
    table.append(el('dt', {}, label), el('dd', {}, String(value)));
  }
  body.append(table);
}

/* -------------------------------------------------------- measurements */

function renderMeasurements(measurements, { pending }) {
  const host = clear($('#measure-list'));
  const pendingNote = $('#measure-pending');
  if (pendingNote) pendingNote.hidden = !pending;

  if (!measurements.length) {
    host.append(el('p.hint', {}, 'No measurements yet.'));
    $('#measure-clear').disabled = true;
    return;
  }
  $('#measure-clear').disabled = false;

  for (const [index, measurement] of measurements.entries()) {
    host.append(el('div.measure-row', {},
      el('span.measure-index', {}, `#${index + 1}`),
      el('span.measure-value', {},
        `${measurement.distance_m.toFixed(2)} m · ${(measurement.distance_m * 3.28084).toFixed(2)} ft`),
      el('button.btn.tiny.danger', {
        type: 'button',
        'aria-label': `Delete measurement ${index + 1}`,
        onclick: () => scene?.removeMeasurement(measurement.id),
      }, '×'),
    ));
  }
}

/* ------------------------------------------------------- frame preview */

function showFrame(view) {
  if (!view || !state.scanId) return;
  const dialog = $('#frame-dialog');
  const image = $('#frame-image');
  image.src = api.frameUrl(state.scanId, view.image_file);
  image.alt = `Captured photo from frame ${view.frame_id}`;
  $('#frame-title').textContent = `Frame ${view.frame_id}`;
  $('#frame-meta').textContent =
    `Camera at x ${view.position[0]}, y ${view.position[1]}, z ${view.position[2]} m`;
  image.onerror = () => {
    $('#frame-meta').textContent = 'This frame image is not available for this scan.';
  };
  dialog.showModal();
}

/* -------------------------------------------------------------- create */

async function createSample() {
  try {
    const scan = await api.createSample('Sample room scan');
    toast('Sample scan queued.', 'success');
    state.scanId = scan.id;
    await refreshScans();
    renderPending(scan);
  } catch (err) {
    toast(err.message, 'error', 8000);
  }
}

async function uploadBundle(files) {
  const list = Array.from(files);
  const cloud = list.find((f) => f.name.toLowerCase().endsWith('.ply'));
  if (!cloud) {
    toast('Include a .ply point cloud in the upload.', 'error', 7000);
    return;
  }
  const poses = list.find((f) => f.name.toLowerCase().endsWith('.json'));
  const frames = list.filter((f) => /\.(jpe?g|png)$/i.test(f.name));

  const bar = $('#upload-progress');
  const barFill = $('#upload-progress-bar');
  bar.hidden = false;
  barFill.style.width = '2%';

  try {
    const scan = await api.createUpload({
      cloud,
      poses,
      frames,
      name: cloud.name.replace(/\.ply$/i, ''),
      onProgress: (ratio) => { barFill.style.width = `${Math.max(2, Math.round(ratio * 100))}%`; },
    });
    toast(`Uploaded ${fmt.bytes(cloud.size)} — processing started.`, 'success');
    state.scanId = scan.id;
    await refreshScans();
    renderPending(scan);
  } catch (err) {
    toast(err.message, 'error', 9000);
  } finally {
    bar.hidden = true;
    barFill.style.width = '0%';
    $('#file-input').value = '';
  }
}

/* ------------------------------------------------------------- controls */

function syncLayerToggles() {
  for (const input of document.querySelectorAll('[data-layer]')) {
    const name = input.dataset.layer;
    if (name === 'grid3d') input.checked = true;
    else if (name === 'pins3d') input.checked = true;
    else input.checked = Boolean(plan?.layers[name]);
  }
}

function wireStaticControls() {
  $('#btn-sample').addEventListener('click', createSample);
  $('#btn-upload').addEventListener('click', () => $('#file-input').click());
  $('#file-input').addEventListener('change', (event) => {
    if (event.target.files?.length) uploadBundle(event.target.files);
  });

  // Drag and drop onto the whole workspace.
  const dropZone = $('#workspace');
  for (const type of ['dragenter', 'dragover']) {
    dropZone.addEventListener(type, (event) => {
      event.preventDefault();
      dropZone.classList.add('is-dropping');
    });
  }
  for (const type of ['dragleave', 'drop']) {
    dropZone.addEventListener(type, (event) => {
      event.preventDefault();
      if (type === 'dragleave' && dropZone.contains(event.relatedTarget)) return;
      dropZone.classList.remove('is-dropping');
    });
  }
  dropZone.addEventListener('drop', (event) => {
    if (event.dataTransfer?.files?.length) uploadBundle(event.dataTransfer.files);
  });

  for (const button of document.querySelectorAll('[data-mode]')) {
    button.addEventListener('click', () => scene?.setMode(button.dataset.mode));
  }
  $('#btn-fit').addEventListener('click', () => plan?.fit());
  $('#btn-zoom-in').addEventListener('click', () => plan?.zoomBy(1.3));
  $('#btn-zoom-out').addEventListener('click', () => plan?.zoomBy(1 / 1.3));
  $('#btn-wireframe').addEventListener('click', (event) => {
    const enabled = !scene?.wireframe;
    scene?.setWireframe(enabled);
    event.currentTarget.classList.toggle('is-active', enabled);
    event.currentTarget.setAttribute('aria-pressed', enabled ? 'true' : 'false');
  });
  $('#btn-recenter').addEventListener('click', () => scene?.frameDollhouse());
  $('#btn-snapshot').addEventListener('click', () => {
    if (!scene || !state.scan) return;
    const link = el('a', { href: scene.snapshot(), download: `${state.scan.id}-view.png` });
    link.click();
    toast('Saved the current 3D view as a PNG.', 'success');
  });
  $('#measure-clear').addEventListener('click', () => scene?.clearMeasurements());

  for (const input of document.querySelectorAll('[data-layer]')) {
    input.addEventListener('change', () => {
      const name = input.dataset.layer;
      if (name === 'grid3d') scene?.setGridVisible(input.checked);
      else if (name === 'pins3d') scene?.setPinsVisible(input.checked);
      else plan?.setLayer(name, input.checked);
    });
  }

  // Export menu (details/summary needs no JS; this only closes it on pick).
  $('#export-menu').addEventListener('click', () => $('#export-details').removeAttribute('open'));

  $('#frame-close').addEventListener('click', () => $('#frame-dialog').close());

  for (const button of document.querySelectorAll('[data-pane]')) {
    button.addEventListener('click', () => {
      const pane = button.dataset.pane;
      document.body.dataset.pane = pane;
      for (const other of document.querySelectorAll('[data-pane]')) {
        const active = other.dataset.pane === pane;
        other.classList.toggle('is-active', active);
        other.setAttribute('aria-pressed', active ? 'true' : 'false');
      }
      // The canvas was hidden, so its size is stale.
      window.dispatchEvent(new Event('resize'));
    });
  }

  $('#btn-library').addEventListener('click', () => {
    const open = document.body.classList.toggle('library-open');
    $('#btn-library').setAttribute('aria-expanded', open ? 'true' : 'false');
  });

  document.addEventListener('keydown', (event) => {
    if (event.target instanceof HTMLElement && /input|textarea|select/i.test(event.target.tagName)) return;
    if (event.key === '1') scene?.setMode(MODE.DOLLHOUSE);
    if (event.key === '2') scene?.setMode(MODE.WALK);
    if (event.key === '3') scene?.setMode(MODE.MEASURE);
    if (event.key === 'f') plan?.fit();
    if (event.key === 'Escape' && document.body.classList.contains('library-open')) {
      document.body.classList.remove('library-open');
    }
  });
}

function setConnection(kind, label) {
  const node = $('#connection');
  node.className = `connection ${kind}`;
  $('#connection-label').textContent = label;
}

boot();
