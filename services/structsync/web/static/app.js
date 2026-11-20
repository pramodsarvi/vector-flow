"use strict";
/* StructuralSync single-page UI: no build step, no framework. Hash routes:
   #/projects   #/project/<id>   #/catalog                                          */

const $ = (sel, el = document) => el.querySelector(sel);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const money = (v) => "$" + Math.round(v).toLocaleString();
const view = $("#view");
const S = { config: null, catalog: [], health: null, token: localStorage.getItem("ss_token") || "" };

// ------------------------------------------------------------------ API
function url(path, params = {}) {
  const u = new URL(path, location.origin);
  for (const [k, v] of Object.entries(params)) if (v !== undefined && v !== null) u.searchParams.set(k, v);
  if (S.token) u.searchParams.set("token", S.token);
  return u.pathname + u.search;
}
async function api(path, opts = {}) {
  const headers = { ...(opts.headers || {}) };
  if (S.token) headers.Authorization = "Bearer " + S.token;
  if (opts.json !== undefined) { headers["Content-Type"] = "application/json"; opts.body = JSON.stringify(opts.json); }
  const res = await fetch(path, { ...opts, headers });
  if (res.status === 401) {
    const t = prompt("This server needs an access token:");
    if (t) { S.token = t; localStorage.setItem("ss_token", t); return api(path, opts); }
  }
  const data = res.headers.get("content-type")?.includes("json") ? await res.json() : null;
  if (!res.ok) throw new Error(data?.detail ? (typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail)) : res.statusText);
  return data;
}
function toast(msg, err = false) {
  const t = document.createElement("div");
  t.className = "toast" + (err ? " err" : "");
  t.textContent = msg;
  $("#toasts").appendChild(t);
  setTimeout(() => t.remove(), err ? 7000 : 3500);
}
const guard = (fn) => async (...a) => { try { await fn(...a); } catch (e) { toast(e.message, true); } };

// ------------------------------------------------------------------ boot + routing
async function boot() {
  try {
    [S.config, S.catalog] = await Promise.all([api("/api/config"), api("/api/catalog")]);
    refreshHealth();
  } catch (e) { toast(e.message, true); }
  document.querySelectorAll("nav button").forEach((b) => b.onclick = () => location.hash = "#/" + b.dataset.view);
  window.onhashchange = route;
  route();
}
async function refreshHealth() {
  try {
    S.health = await api("/api/health");
    const h = S.health;
    $("#status").innerHTML = `<span class="dot ok"></span><span>${esc(h.gpu || h.device)}</span>`;
  } catch { $("#status").innerHTML = `<span class="dot bad"></span><span>offline</span>`; }
}
let pollTimer = null;
function route() {
  clearTimeout(pollTimer);
  const [, page, id] = location.hash.split("/");
  document.querySelectorAll("nav button").forEach((b) =>
    b.classList.toggle("active", b.dataset.view === (page === "catalog" ? "catalog" : "projects")));
  if (page === "project" && id) return projectView(id);
  if (page === "catalog") return catalogView();
  return projectsView();
}

// ------------------------------------------------------------------ projects list
async function projectsView() {
  view.innerHTML = `
    <h1>Projects</h1><p class="sub">Upload a kitchen photo. Surfaces are detected automatically (≈10–30 s), then you can try materials.</p>
    <div class="upload" id="drop">
      <div class="grow"><b>New project</b><div class="muted">Drop a photo here or choose a file (JPG/PNG, ≤25 MB)</div></div>
      <input type="text" id="pname" placeholder="Project name (optional)" style="width:220px">
      <input type="file" id="pfile" accept="image/*" hidden>
      <button class="btn primary" id="pick">Choose photo</button>
    </div>
    <div class="grid" id="plist"><div class="empty"><span class="spinner"></span></div></div>`;
  const create = guard(async (file) => {
    if (!file) return;
    const fd = new FormData();
    fd.append("photo", file);
    fd.append("name", $("#pname").value);
    $("#pick").disabled = true; $("#pick").innerHTML = `<span class="spinner"></span> Uploading`;
    try {
      const p = await api("/api/projects", { method: "POST", body: fd });
      location.hash = "#/project/" + p.id;
    } finally { $("#pick").disabled = false; $("#pick").textContent = "Choose photo"; }
  });
  $("#pick").onclick = () => $("#pfile").click();
  $("#pfile").onchange = (e) => create(e.target.files[0]);
  const drop = $("#drop");
  drop.ondragover = (e) => { e.preventDefault(); drop.classList.add("drag"); };
  drop.ondragleave = () => drop.classList.remove("drag");
  drop.ondrop = (e) => { e.preventDefault(); drop.classList.remove("drag"); create(e.dataTransfer.files[0]); };

  const list = await api("/api/projects").catch((e) => (toast(e.message, true), []));
  const el = $("#plist");
  if (!list.length) { el.innerHTML = `<div class="empty">No projects yet.</div>`; return; }
  el.innerHTML = list.map((p) => `
    <div class="card" data-id="${esc(p.id)}">
      <div class="thumb" style="background-image:url('${url(`/api/projects/${p.id}/photo`, { size: 520 })}')"></div>
      <div class="body">
        <div class="row"><div class="title grow">${esc(p.name)}</div><span class="badge ${esc(p.status)}">${esc(p.status)}</span></div>
        <div class="row muted" style="font-size:12px"><span class="grow">${new Date(p.created * 1000).toLocaleString()}</span>
          <span>${p.renders} render${p.renders === 1 ? "" : "s"}</span>
          <button class="btn small danger" data-del="${esc(p.id)}">Delete</button></div>
      </div>
    </div>`).join("");
  el.onclick = guard(async (e) => {
    const del = e.target.closest("[data-del]");
    if (del) {
      e.stopPropagation();
      if (!confirm("Delete this project and all its renders?")) return;
      await api(`/api/projects/${del.dataset.del}`, { method: "DELETE" });
      toast("Project deleted");
      return projectsView();
    }
    const card = e.target.closest(".card");
    if (card) location.hash = "#/project/" + card.dataset.id;
  });
}

// ------------------------------------------------------------------ project workspace
async function projectView(id) {
  view.innerHTML = `<div class="empty"><span class="spinner"></span></div>`;
  let p;
  try { p = await api(`/api/projects/${id}`); } catch (e) { toast(e.message, true); location.hash = "#/projects"; return; }
  if (!S.health) await refreshHealth();
  S.catalog = await api("/api/catalog").catch(() => S.catalog);   // may have changed (import / delete)
  const st = {
    p, mode: "surfaces", target: S.config.surfaces[0], action: "add", busy: false,
    assign: Object.fromEntries(Object.entries(p.assign || {}).filter(([, v]) => S.catalog.some((s) => s.id === v))),
    current: p.renders[0] || null,
    opts: { edges: true, grain: true, reflections: false, material_maps: false, local_tint: false, harmonize: false, strength: 0.3 },
  };
  if (st.current) { st.mode = "result"; Object.assign(st.opts, st.current.options || {}); }

  view.innerHTML = `
    <div class="row" style="margin-bottom:16px">
      <a href="#/projects" class="btn small">← Projects</a>
      <h1 class="grow" id="ptitle" title="Click to rename" style="cursor:text">${esc(p.name)}</h1>
      <span class="badge" id="pstatus"></span>
      <button class="btn small" id="reanalyze">Re-analyse</button>
    </div>
    <div class="workspace">
      <div style="display:flex;flex-direction:column;gap:14px;min-width:0">
        <div class="row">
          <div class="tabs" id="modes">
            <button data-mode="surfaces">Surfaces</button><button data-mode="result">Before / after</button>
          </div>
          <div class="grow"></div>
          <div class="row" id="correct-bar">
            <span class="muted">Click the photo to</span>
            <div class="tabs" id="actions"><button data-a="add">add to</button><button data-a="remove">remove from</button></div>
            <select id="ctarget"></select>
          </div>
        </div>
        <div id="canvas"></div>
        <div class="panel" id="result-panel"></div>
        <div class="panel"><h2>Render history</h2><div class="history" id="history"></div></div>
      </div>
      <div class="side">
        <div class="panel"><h2>Materials</h2><div id="materials"></div></div>
        <div class="panel"><h2>Options</h2><div id="options"></div>
          <button class="btn primary" id="render" style="width:100%;margin-top:12px">Render</button></div>
      </div>
    </div>`;

  $("#ptitle").onclick = guard(async () => {
    const n = prompt("Project name", st.p.name);
    if (!n) return;
    st.p = await api(`/api/projects/${id}`, { method: "PATCH", json: { name: n } });
    $("#ptitle").textContent = st.p.name;
  });
  $("#reanalyze").onclick = guard(async () => {
    if (!confirm("Re-run detection? Manual corrections will be lost.")) return;
    st.p = await api(`/api/projects/${id}/reanalyze`, { method: "POST" });
    draw();
  });
  $("#modes").onclick = (e) => { const b = e.target.closest("button"); if (b) { st.mode = b.dataset.mode; draw(); } };
  $("#actions").onclick = (e) => { const b = e.target.closest("button"); if (b) { st.action = b.dataset.a; draw(); } };
  $("#ctarget").onchange = (e) => { st.target = e.target.value; };
  $("#render").onclick = guard(render);

  // A region id is "<kind>#<n>". The API also accepts a bare kind, which means "all regions of
  // this kind" when removing and "a new region" when adding — the case where nothing was detected.
  const kindOf = (key) => String(key).split("#")[0];
  function regionList() { return st.p.regions || []; }
  function regionsOf(kind) { return regionList().filter((r) => r.kind === kind); }
  function regionLabel(r) {
    const n = regionsOf(r.kind).length;
    return n > 1 ? `${r.kind} ${regionsOf(r.kind).indexOf(r) + 1}` : r.kind;
  }
  const rgb = (c) => (Array.isArray(c) ? `rgb(${c.join(",")})` : c);
  function regionColor(r) { return rgb(r.color) || S.config.colors[r.kind]; }
  // a material chosen for a whole class applies to each of its regions
  function assignedTo(r) { return st.assign[r.id] ?? st.assign[r.kind] ?? ""; }

  function drawCorrectTarget() {
    const sel = $("#ctarget");
    if (!sel) return;
    const opts = [];
    for (const k of S.config.surfaces) {
      for (const r of regionsOf(k)) opts.push([r.id, regionLabel(r)]);
      opts.push([k, regionsOf(k).length ? `+ new ${k}` : k]);
    }
    if (!opts.some(([v]) => v === st.target)) st.target = opts[0]?.[0] ?? S.config.surfaces[0];
    sel.innerHTML = opts.map(([v, t]) =>
      `<option value="${esc(v)}" ${v === st.target ? "selected" : ""}>${esc(t)}</option>`).join("");
  }

  function draw() {
    const ready = st.p.status === "ready";
    $("#pstatus").className = "badge " + st.p.status;
    $("#pstatus").textContent = st.p.status;
    document.querySelectorAll("#modes button").forEach((b) => b.classList.toggle("active", b.dataset.mode === st.mode));
    document.querySelectorAll("#actions button").forEach((b) => b.classList.toggle("active", b.dataset.a === st.action));
    $("#correct-bar").style.visibility = st.mode === "surfaces" && ready ? "visible" : "hidden";
    $("#render").disabled = !ready || st.busy;
    drawCorrectTarget();
    drawCanvas(); drawMaterials(); drawOptions(); drawResult(); drawHistory();
    if (st.p.status === "analyzing") pollTimer = setTimeout(poll, 2000);
  }
  async function poll() {
    try { st.p = await api(`/api/projects/${id}`); } catch { /* transient */ }
    if (location.hash !== "#/project/" + id) return;
    if (st.p.status !== "analyzing") {
      toast(st.p.status === "ready" ? (st.p.analysis_cached ? "Same photo analysed before — reused" : "Surfaces detected") : "Analysis failed", st.p.status !== "ready");
      draw();
    }
    else pollTimer = setTimeout(poll, 2000);
  }

  function drawCanvas() {
    const c = $("#canvas");
    if (st.p.status === "analyzing") {
      c.innerHTML = `<div class="stage"><img src="${url(`/api/projects/${id}/photo`)}" alt="">
        <div class="busy-veil"><span class="spinner"></span>Detecting surfaces and geometry…</div></div>`;
      return;
    }
    if (st.p.status === "error") {
      c.innerHTML = `<div class="stage"><img src="${url(`/api/projects/${id}/photo`)}" alt=""></div>
        <p class="note">Analysis failed: ${esc(st.p.error)}</p>`;
      return;
    }
    if (st.mode === "result" && st.current) {
      c.innerHTML = `<div class="compare" id="cmp" style="--pos:50%">
          <img src="${url(`/api/projects/${id}/photo`, { full: 1 })}" alt="before">
          <div class="after"><img src="${url(`/api/projects/${id}/renders/${st.current.id}.jpg`)}" alt="after"></div>
          <span class="lbl" style="left:10px">BEFORE</span><span class="lbl" style="right:10px">AFTER</span>
          <div class="bar"></div><div class="knob">⇆</div>
          <input type="range" min="0" max="100" value="50" aria-label="Before/after position"></div>`;
      $("#cmp input").oninput = (e) => $("#cmp").style.setProperty("--pos", e.target.value + "%");
      return;
    }
    if (st.mode === "result") { c.innerHTML = `<div class="panel empty">Pick materials and press <b>Render</b>.</div>`; return; }
    c.innerHTML = `<div class="stage correcting" id="stage"><img id="ov" src="${url(`/api/projects/${id}/overlay`, { v: Date.now() })}" alt="detected surfaces"></div>
      <div class="row" style="margin-top:8px">${(regionList().length
          ? regionList().map((r) => [regionLabel(r), regionColor(r)])
          : Object.entries(S.config.colors).map(([k, col]) => [k, col])
        ).map(([label, col]) =>
        `<span class="row" style="gap:5px;font-size:12px"><span class="chip" style="background:${col}"></span>${esc(label)}</span>`).join("")}</div>`;
    $("#ov").onclick = guard(async (e) => {
      if (st.busy) return;
      const img = e.target, r = img.getBoundingClientRect();
      // overlay is served at analysis resolution: scale display px → natural px
      const x = (e.clientX - r.left) * img.naturalWidth / r.width, y = (e.clientY - r.top) * img.naturalHeight / r.height;
      const ping = document.createElement("div");
      ping.className = "ping"; ping.style.left = (e.clientX - r.left) + "px"; ping.style.top = (e.clientY - r.top) + "px";
      $("#stage").appendChild(ping);
      st.busy = true;
      const label = st.target.includes("#") ? st.target.replace("#", " ") : st.target;
      const veil = document.createElement("div"); veil.className = "busy-veil"; veil.innerHTML = `<span class="spinner"></span>Updating ${esc(label)}…`;
      $("#stage").appendChild(veil);
      try {
        st.p = await api(`/api/projects/${id}/correct`, {
          method: "POST",
          json: { kind: kindOf(st.target), region: st.target.includes("#") ? st.target : null,
                  x, y, mode: st.action },
        });
      } finally { st.busy = false; draw(); }
    });
  }

  function drawMaterials() {
    const el = $("#materials");
    const byKind = Object.fromEntries((st.p.surfaces || []).map((s) => [s.kind, s]));
    // One row per addressable region, so two walls or two cabinet doors can take different
    // materials. A project analysed before regions existed has no region list: it falls back to a
    // row per class, which is exactly what it showed before.
    const rows = regionList().length
      ? regionList().map((r) => ({ key: r.id, kind: r.kind, label: regionLabel(r),
                                   area: r.area_m2, color: regionColor(r), found: true,
                                   sel: assignedTo(r) }))
      : S.config.surfaces.map((k) => ({ key: k, kind: k, label: k, area: byKind[k]?.area_m2,
                                        color: S.config.colors[k], found: !!byKind[k],
                                        sel: st.assign[k] || "" }));
    el.innerHTML = rows.map((row) => {
      const opts = S.catalog.filter((s) => s.category === row.kind);
      const price = (S.catalog.find((x) => x.id === row.sel) || {}).price || 0;
      return `<div class="surface-row">
        <span class="chip" style="background:${row.color}"></span>
        <div class="surface-name">${esc(row.label)}<small>${row.found ? `≈${row.area} m²` : "not detected"}${row.sel ? ` · ${money(price)}/m²` : ""}</small></div>
        <select data-k="${esc(row.key)}" ${row.found && opts.length ? "" : "disabled"}>
          <option value="">Keep original</option>
          ${opts.map((s) => `<option value="${esc(s.id)}" ${s.id === row.sel ? "selected" : ""}>${esc(s.name)}</option>`).join("")}
        </select>
        <div class="swatch" style="${row.sel ? `background-image:url('${url(`/api/catalog/${row.sel}/albedo`, { size: 96 })}')` : ""}"></div>
      </div>`;
    }).join("") + `<div class="row" style="margin-top:10px">
        <input type="text" id="q" class="grow" placeholder="Describe a material, e.g. dark marble">
        <button class="btn" id="qgo">Find</button></div>`;
    el.querySelectorAll("select").forEach((s) => s.onchange = () => {
      const key = s.dataset.k;
      // a row is now a region: setting one drops the class-wide value it was inheriting, so the
      // two can't disagree about the same pixels
      if (key.includes("#")) delete st.assign[kindOf(key)];
      st.assign[key] = s.value || null;
      drawMaterials();
    });
    const find = guard(async () => {
      const q = $("#q").value.trim();
      if (!q) return;
      const targets = rows.filter((r) => r.found && S.catalog.some((s) => s.category === r.kind));
      const res = await Promise.all(targets.map((r) => api(url("/api/catalog/search", { q, category: r.kind, k: 1 }))));
      const best = res.map((r, i) => [targets[i], r[0]]).filter(([, h]) => h).sort((a, b) => b[1].score - a[1].score)[0];
      if (!best) return toast("No matching material");
      const [row, hit] = best;
      if (row.key.includes("#")) delete st.assign[kindOf(row.key)];
      st.assign[row.key] = hit.id;
      toast(`${hit.name} → ${row.label}`);
      drawMaterials();
    });
    $("#qgo").onclick = find;
    $("#q").onkeydown = (e) => { if (e.key === "Enter") find(); };
  }

  function drawOptions() {
    const hw = S.health?.harmonizer || {};
    const canHarm = hw.weights_present && !hw.error;
    const items = [
      ["edges", "Countertop front edges", "Continue the top material over the slab edge"],
      ["grain", "Match camera grain", "Add the photo's sensor noise to new materials"],
      ["reflections", "Mirror reflections", "Experimental; polished stone only"],
      ["material_maps", "Surface relief", "Experimental; uses normal/roughness maps (tiles, wood)"],
      ["local_tint", "Local light colour", "Experimental; tint from neutral surroundings"],
      ["harmonize", "SDXL seam harmonisation", canHarm ? "Slow; seams only" : "Unavailable: weights not installed"],
    ];
    $("#options").innerHTML = items.map(([k, t, d]) => `<div class="toggle"><div>${t}<small>${d}</small></div>
      <label class="switch"><input type="checkbox" data-o="${k}" ${st.opts[k] ? "checked" : ""} ${k === "harmonize" && !canHarm ? "disabled" : ""}><span></span></label></div>`).join("")
      + (st.opts.harmonize ? `<label class="field" style="margin-top:6px">Strength ${st.opts.strength}
          <input type="range" id="strength" min="0.1" max="0.5" step="0.05" value="${st.opts.strength}"></label>` : "");
    $("#options").querySelectorAll("input[data-o]").forEach((i) => i.onchange = () => { st.opts[i.dataset.o] = i.checked; drawOptions(); });
    const s = $("#strength");
    if (s) s.oninput = () => { st.opts.strength = +s.value; s.previousSibling.textContent = "Strength " + s.value; };
  }

  async function render() {
    const assign = Object.fromEntries(Object.entries(st.assign).filter(([, v]) => v));
    if (!Object.keys(assign).length) return toast("Pick at least one material", true);
    st.busy = true;
    $("#render").disabled = true; $("#render").innerHTML = `<span class="spinner"></span> Rendering…`;
    try {
      const rec = await api(`/api/projects/${id}/render`, { method: "POST", json: { assign, ...st.opts } });
      st.p.renders = [rec, ...st.p.renders.filter((x) => x.id !== rec.id)];   // a cache hit may return an existing render
      st.current = rec; st.mode = "result";
      if (rec.cached) toast("Same scene and style rendered before — reused instantly");
      if (rec.report?.note) toast(rec.report.note, true);
    } finally { st.busy = false; $("#render").textContent = "Render"; draw(); }
  }

  function drawResult() {
    const el = $("#result-panel");
    const r = st.current;
    if (!r || st.mode !== "result") { el.style.display = "none"; return; }
    el.style.display = "";
    const rep = r.report || {};
    const sku = Object.entries(rep.sku_check || {});
    el.innerHTML = `
      <div class="row" style="margin-bottom:12px"><h2 class="grow" style="margin:0">Bill of materials</h2>
        <a class="btn small" href="${url(`/api/projects/${id}/renders/${r.id}.jpg`)}" download="render-${r.id}.jpg">Download image</a></div>
      <table><thead><tr><th>Surface</th><th>Product</th><th>Vendor</th><th class="num">Area m²</th><th class="num">Est. cost</th></tr></thead>
        <tbody>${r.bom.map((b) => `<tr><td style="text-transform:capitalize">${esc(b.surface)}</td><td>${esc(b.product)} <span class="muted">${esc(b.sku)}</span></td>
          <td>${esc(b.vendor)}</td><td class="num">${esc(b.range)}</td><td class="num">${money(b.est_cost)}</td></tr>`).join("")}
        <tr><td colspan="4"><b>Total</b> <span class="muted">(±20 %, verify on site)</span></td><td class="num"><b>${money(r.total)}</b></td></tr></tbody></table>
      <h2 style="margin-top:18px">Checks</h2>
      <div class="checks">
        <div class="check"><b>${(rep.outside_change ?? 0) < 0.001 ? "✓" : rep.outside_change}</b><span>outside edit untouched</span></div>
        <div class="check"><b>${Math.round((rep.boundary_edge_recall ?? 0) * 100)}%</b><span>structural edges kept</span></div>
        ${sku.map(([k, c]) => `<div class="check"><b>${c.pass ? "✓" : "#" + c.rank}</b><span>${esc(k)} recognised</span></div>`).join("")}
        <div class="check"><b>${r.cached ? "cached" : r.seconds + "s"}</b><span>${r.cached ? "reused, not recomputed" : "render time"}</span></div>
      </div>
      ${rep.note ? `<p class="note" style="margin-top:12px">${esc(rep.note)}</p>` : ""}`;
  }

  function drawHistory() {
    const el = $("#history");
    if (!st.p.renders.length) { el.innerHTML = `<span class="muted">Renders you make appear here.</span>`; return; }
    el.innerHTML = st.p.renders.map((r) => `<div class="h ${st.current && r.id === st.current.id ? "active" : ""}" data-r="${r.id}">
      <img loading="lazy" src="${url(`/api/projects/${id}/renders/${r.id}.jpg`, { size: 360 })}" alt="">
      <div class="meta">${Object.values(r.assign).length} materials · ${money(r.total)}${r.cached ? " · cached" : ""}</div>
      <button class="x" data-x="${r.id}" title="Delete render">×</button></div>`).join("");
    el.onclick = guard(async (e) => {
      const x = e.target.closest("[data-x]");
      if (x) {
        if (!confirm("Delete this render?")) return;
        await api(`/api/projects/${id}/renders/${x.dataset.x}`, { method: "DELETE" });
        st.p.renders = st.p.renders.filter((r) => r.id !== x.dataset.x);
        if (st.current?.id === x.dataset.x) st.current = st.p.renders[0] || null;
        return draw();
      }
      const h = e.target.closest("[data-r]");
      if (!h) return;
      st.current = st.p.renders.find((r) => r.id === h.dataset.r);
      st.assign = Object.fromEntries(Object.entries(st.current.assign).filter(([, v]) => S.catalog.some((s) => s.id === v)));
      Object.assign(st.opts, st.current.options || {});
      st.mode = "result";
      draw();
    });
  }
  draw();
}

// ------------------------------------------------------------------ catalogue
let catFilter = "all";
async function catalogView() {
  S.catalog = await api("/api/catalog").catch((e) => (toast(e.message, true), S.catalog));
  const cats = ["all", ...S.config.surfaces];
  const items = S.catalog.filter((s) => catFilter === "all" || s.category === catFilter);
  view.innerHTML = `
    <div class="row" style="margin-bottom:4px"><h1 class="grow">Catalogue</h1>
      <a class="btn" href="${url("/api/catalog/export")}" download>Export</a>
      <button class="btn" id="import">Import ZIP…</button>
      <button class="btn primary" id="add">+ Add material</button></div>
    <p class="sub">${S.catalog.length} materials. Textures are seamless tiles; tile size sets their real-world scale.</p>
    <div class="filters">${cats.map((c) => `<button data-c="${c}" class="${c === catFilter ? "active" : ""}">${c}
      <span class="muted">${c === "all" ? S.catalog.length : S.catalog.filter((s) => s.category === c).length}</span></button>`).join("")}</div>
    <div class="grid">${items.map((s) => `
      <div class="card sku" style="cursor:default">
        <div class="thumb" style="background-image:url('${url(`/api/catalog/${s.id}/albedo`, { size: 400, v: S.ver || 0 })}')"></div>
        <div class="body">
          <div class="row"><div class="title grow">${esc(s.name)}</div><span class="badge">${esc(s.category)}</span></div>
          ${s.has_normal || s.has_roughness ? `<div class="row" style="gap:4px">${s.has_normal ? '<span class="badge pass">normal map</span>' : ""}${s.has_roughness ? '<span class="badge pass">roughness</span>' : ""}</div>` : ""}
          <div class="specs"><span>SKU <b>${esc(s.id)}</b></span><span>Vendor <b>${esc(s.vendor)}</b></span>
            <span>Price <b>${money(s.price)}/m²</b></span><span>Tile <b>${s.tile_w}×${s.tile_h} m</b></span>
            <span>Gloss <b>${s.gloss}</b></span></div>
          <div class="actions"><button class="btn small" data-edit="${esc(s.id)}">Edit</button>
            <button class="btn small danger" data-del="${esc(s.id)}">Delete</button></div>
        </div>
      </div>`).join("") || `<div class="empty">No materials in this category.</div>`}</div>`;
  view.querySelector(".filters").onclick = (e) => { const b = e.target.closest("[data-c]"); if (b) { catFilter = b.dataset.c; catalogView(); } };
  $("#add").onclick = () => skuDialog(null);
  $("#import").onclick = importDialog;
  view.querySelector(".grid").onclick = guard(async (e) => {
    const ed = e.target.closest("[data-edit]");
    if (ed) return skuDialog(S.catalog.find((s) => s.id === ed.dataset.edit));
    const del = e.target.closest("[data-del]");
    if (del) {
      if (!confirm(`Delete ${del.dataset.del}? Existing renders keep their images.`)) return;
      await api(`/api/catalog/${del.dataset.del}`, { method: "DELETE" });
      toast("Material deleted");
      catalogView();
    }
  });
}

function skuDialog(sku) {
  const dlg = $("#sku-dialog"), f = $("#sku-form");
  f.reset();
  $("#sku-category").innerHTML = S.config.surfaces.map((k) => `<option value="${k}">${k}</option>`).join("");
  $("#sku-title").textContent = sku ? "Edit material" : "Add material";
  $("#tex-label").textContent = sku ? "Replace texture (optional)" : "Texture (PNG/JPG, seamless)";
  $("#maps-label").textContent = sku && (sku.has_normal || sku.has_roughness)
    ? `Has ${[sku.has_normal && "normal", sku.has_roughness && "roughness"].filter(Boolean).join(" + ")} map — upload to replace`
    : "Optional relief maps (same tiling as the texture)";
  f.id.disabled = !!sku;
  f.albedo.required = !sku;
  if (sku) for (const k of ["id", "name", "category", "vendor", "price", "tile_w", "tile_h", "gloss"]) f[k].value = sku[k];
  else if (catFilter !== "all") f.category.value = catFilter;
  $("#gloss-val").textContent = (+f.gloss.value).toFixed(2);
  f.gloss.oninput = () => $("#gloss-val").textContent = (+f.gloss.value).toFixed(2);
  const prev = $("#tex-preview");
  prev.style.backgroundImage = sku ? `url('${url(`/api/catalog/${sku.id}/albedo`, { size: 600, v: S.ver || 0 })}')` : "none";
  f.albedo.onchange = () => { const file = f.albedo.files[0]; if (file) prev.style.backgroundImage = `url('${URL.createObjectURL(file)}')`; };
  $("#sku-save").onclick = guard(async (e) => {
    e.preventDefault();
    if (!f.reportValidity()) return;
    $("#sku-save").disabled = true;
    try {
      if (sku) {
        const body = Object.fromEntries(["name", "category", "vendor", "price", "tile_w", "tile_h", "gloss"].map((k) =>
          [k, ["price", "tile_w", "tile_h", "gloss"].includes(k) ? +f[k].value : f[k].value]));
        await api(`/api/catalog/${sku.id}`, { method: "PUT", json: body });
        if (f.albedo.files[0]) {
          const fd = new FormData(); fd.append("albedo", f.albedo.files[0]);
          await api(`/api/catalog/${sku.id}/albedo`, { method: "POST", body: fd });
        }
        if (f.normal.files[0] || f.roughness.files[0]) {
          const fd = new FormData();
          if (f.normal.files[0]) fd.append("normal", f.normal.files[0]);
          if (f.roughness.files[0]) fd.append("roughness", f.roughness.files[0]);
          await api(`/api/catalog/${sku.id}/maps`, { method: "POST", body: fd });
        }
      } else {
        const fd = new FormData(f);
        await api("/api/catalog", { method: "POST", body: fd });
      }
      S.ver = Date.now();
      dlg.close();
      toast(sku ? "Material updated" : "Material added");
      S.catalog = await api("/api/catalog");
      catalogView();
    } finally { $("#sku-save").disabled = false; }
  });
  dlg.showModal();
}

boot();

// ------------------------------------------------------------------ bulk import
function importDialog() {
  const dlg = $("#import-dialog"), f = $("#import-form"), out = $("#import-result");
  f.reset(); out.innerHTML = ""; $("#import-apply").disabled = true;
  $("#tpl-link").href = url("/api/catalog/export"); $("#tpl-link").setAttribute("download", "");
  const list = (ids, cls = "") => `<ul class="import-list ${cls}">${ids.map((x) => `<li>${esc(x)}</li>`).join("")}</ul>`;
  const send = async (dry) => {
    const fd = new FormData();
    fd.append("archive", f.archive.files[0]);
    fd.append("mode", f.mode.value);
    fd.append("dry_run", dry ? "true" : "false");
    return api("/api/catalog/import", { method: "POST", body: fd });
  };
  const show = (r) => {
    out.innerHTML = r.ok
      ? `<p><b>${r.applied ? "Imported" : "Ready to import"}:</b> ${r.added.length} new, ${r.updated.length} updated` +
        (r.mode === "replace" ? `, <span class="danger-text">${r.removed.length} removed</span>` : "") + ` → ${r.total} materials.</p>` +
        (r.removed.length ? `<p class="danger-text">These materials will be deleted${r.applied ? "" : ` — <a href="${url("/api/catalog/export")}" download>export the current catalogue first</a> if you may need them`}:</p>` + list(r.removed) : "") +
        (r.warnings.length ? `<p class="muted">Notes:</p>` + list(r.warnings) : "")
      : `<p class="danger-text"><b>${r.error_count} problem${r.error_count === 1 ? "" : "s"} — nothing was changed.</b></p>` + list(r.errors, "danger-text") +
        (r.error_count > r.errors.length ? `<p class="muted">…and ${r.error_count - r.errors.length} more</p>` : "");
  };
  f.archive.onchange = () => { out.innerHTML = ""; $("#import-apply").disabled = true; };
  f.querySelectorAll("input[name=mode]").forEach((i) => i.onchange = () => { out.innerHTML = ""; $("#import-apply").disabled = true; });
  $("#import-check").onclick = guard(async (e) => {
    e.preventDefault();
    if (!f.archive.files[0]) return toast("Choose a ZIP file first", true);
    $("#import-check").disabled = true; $("#import-check").innerHTML = `<span class="spinner"></span> Checking…`;
    try { const r = await send(true); show(r); $("#import-apply").disabled = !r.ok; }
    finally { $("#import-check").disabled = false; $("#import-check").textContent = "Check file"; }
  });
  $("#import-apply").onclick = guard(async (e) => {
    e.preventDefault();
    $("#import-apply").disabled = true; $("#import-apply").innerHTML = `<span class="spinner"></span> Importing…`;
    try {
      const r = await send(false);
      show(r);
      if (r.applied) { toast(`Catalogue updated: ${r.total} materials`); S.ver = Date.now(); S.catalog = await api("/api/catalog"); catalogView(); }
    } finally {
      // Re-enable as well as relabel: a failed import previously left the
      // button permanently disabled with no way to retry.
      $("#import-apply").textContent = "Import";
      $("#import-apply").disabled = false;
    }
  });
  dlg.showModal();
}
