/**
 * Interactive 2D floor plan.
 *
 * Drawn from floorplan.json rather than shown as the pre-rendered SVG, so the
 * plan can be panned, zoomed, hovered and selected, and so the 2D and 3D views
 * read from the same single source of truth. The server's floorplan.svg
 * remains the thing you download for print and CAD hand-off.
 *
 * Plan coordinates are metres: x to the right, z into the screen. The SVG user
 * space mirrors that 1:1 (1 unit = 1 m) and all scaling is done with viewBox,
 * so geometry never has to be re-projected when the user zooms.
 */

const NS = 'http://www.w3.org/2000/svg';
const MIN_ZOOM = 0.25;
const MAX_ZOOM = 12;

function svg(tag, attrs = {}) {
  const node = document.createElementNS(NS, tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (value !== null && value !== undefined) node.setAttribute(key, String(value));
  }
  return node;
}

export class Plan2D {
  /**
   * @param {HTMLElement} container
   * @param {{onSelect?: (selection: object|null) => void}} options
   */
  constructor(container, { onSelect } = {}) {
    this.container = container;
    this.onSelect = onSelect || (() => {});
    this.floorplan = null;
    this.selection = null;
    this.layers = {
      dimensions: true,
      openings: true,
      furniture: true,
      cameras: false,
      grid: true,
    };
    // Current view in plan metres; set by fit().
    this.view = { x: 0, y: 0, w: 10, h: 10 };
    this.bounds = null;

    this.svg = svg('svg', {
      class: 'plan',
      xmlns: NS,
      role: 'img',
      'aria-label': 'Two-dimensional floor plan',
      preserveAspectRatio: 'xMidYMid meet',
    });
    this.root = svg('g');
    this.svg.append(this.root);
    container.append(this.svg);

    this._bindInteraction();
    // A ResizeObserver rather than a window resize listener: the container can
    // still be zero-sized when render() first runs (hidden pane, fonts not yet
    // settled, layout not flushed). Without this the viewBox is never written
    // and the plan draws at the wrong scale.
    this._observer = new ResizeObserver(() => this._applyView());
    this._observer.observe(container);
  }

  destroy() {
    this._observer?.disconnect();
    this.svg.remove();
  }

  /* ------------------------------------------------------------- drawing */

  render(floorplan) {
    this.floorplan = floorplan;
    this.selection = null;
    this.bounds = this._computeBounds(floorplan);
    this.fit();
    this._draw();
  }

  setLayer(name, enabled) {
    this.layers[name] = enabled;
    if (this.floorplan) this._draw();
  }

  _computeBounds(floorplan) {
    const polygon = floorplan?.room?.polygon || [];
    if (!polygon.length) return { minX: 0, minZ: 0, maxX: 1, maxZ: 1 };
    const xs = polygon.map((p) => p.x);
    const zs = polygon.map((p) => p.z);
    return {
      minX: Math.min(...xs),
      maxX: Math.max(...xs),
      minZ: Math.min(...zs),
      maxZ: Math.max(...zs),
    };
  }

  /** Frame the whole room with a margin wide enough for the dimension lines. */
  fit() {
    if (!this.bounds) return;
    const margin = 1.1;
    this.view = {
      x: this.bounds.minX - margin,
      y: this.bounds.minZ - margin,
      w: this.bounds.maxX - this.bounds.minX + margin * 2,
      h: this.bounds.maxZ - this.bounds.minZ + margin * 2,
    };
    this._applyView();
  }

  zoomBy(factor) {
    const cx = this.view.x + this.view.w / 2;
    const cy = this.view.y + this.view.h / 2;
    this._zoomAt(cx, cy, factor);
  }

  _zoomAt(px, py, factor) {
    if (!this.bounds) return;
    const span = Math.max(this.bounds.maxX - this.bounds.minX, this.bounds.maxZ - this.bounds.minZ) || 1;
    const nextW = this.view.w / factor;
    // Clamp against the room size so the user cannot zoom into empty space forever.
    if (nextW < span / MAX_ZOOM || nextW > span / MIN_ZOOM) return;
    const ratio = this.view.h / this.view.w;
    this.view = {
      w: nextW,
      h: nextW * ratio,
      x: px - (px - this.view.x) / factor,
      y: py - (py - this.view.y) / factor,
    };
    this._applyView();
  }

  _applyView() {
    const { width: w, height: h } = this.container.getBoundingClientRect();
    if (!w || !h) {
      this._needsLayout = true;
      return;
    }
    this._needsLayout = false;
    // Letterbox the plan view into the container's aspect ratio so metres stay
    // square — a stretched floor plan is a wrong floor plan.
    const aspect = w / h;
    let vw = this.view.w;
    let vh = this.view.h;
    if (vw / vh > aspect) vh = vw / aspect;
    else vw = vh * aspect;
    const cx = this.view.x + this.view.w / 2;
    const cy = this.view.y + this.view.h / 2;
    this.svg.setAttribute('viewBox', `${cx - vw / 2} ${cy - vh / 2} ${vw} ${vh}`);
    this._scalePx = w / vw; // pixels per metre, for line widths and label sizes
    this._restyleForZoom();
  }

  /** Keep strokes and text a constant on-screen size regardless of zoom. */
  _restyleForZoom() {
    const px = 1 / (this._scalePx || 60);
    this.root.style.setProperty('--px', `${px}`);
    this.root.querySelectorAll('[data-px-stroke]').forEach((node) => {
      node.setAttribute('stroke-width', px * Number(node.dataset.pxStroke));
    });
    this.root.querySelectorAll('[data-px-font]').forEach((node) => {
      node.setAttribute('font-size', px * Number(node.dataset.pxFont));
    });
  }

  _draw() {
    const room = this.floorplan?.room;
    this.root.replaceChildren();
    if (!room) return;

    if (this.layers.grid) this.root.append(this._grid());
    this.root.append(this._floor(room));
    this.root.append(this._walls(room));
    if (this.layers.openings) this.root.append(this._openings(room));
    if (this.layers.furniture) this.root.append(this._furniture(room));
    if (this.layers.dimensions) this.root.append(this._dimensions(room));
    if (this.layers.cameras) this.root.append(this._cameras());
    this._restyleForZoom();
    this._highlightSelection();
  }

  _grid() {
    const group = svg('g', { class: 'layer-grid', 'aria-hidden': 'true' });
    const { minX, maxX, minZ, maxZ } = this.bounds;
    const pad = 1.2;
    for (let x = Math.floor(minX - pad); x <= Math.ceil(maxX + pad); x += 1) {
      const line = svg('line', { x1: x, y1: minZ - pad, x2: x, y2: maxZ + pad, class: 'grid-line' });
      line.dataset.pxStroke = x % 5 === 0 ? 1.2 : 0.6;
      group.append(line);
    }
    for (let z = Math.floor(minZ - pad); z <= Math.ceil(maxZ + pad); z += 1) {
      const line = svg('line', { x1: minX - pad, y1: z, x2: maxX + pad, y2: z, class: 'grid-line' });
      line.dataset.pxStroke = z % 5 === 0 ? 1.2 : 0.6;
      group.append(line);
    }
    return group;
  }

  _floor(room) {
    const points = (room.polygon || []).map((p) => `${p.x},${p.z}`).join(' ');
    const group = svg('g', { class: 'layer-floor' });
    group.append(svg('polygon', { points, class: 'floor-fill' }));
    // Near the top edge rather than dead centre: the centre is where furniture
    // usually sits, and the two labels collided.
    const label = svg('text', {
      x: (this.bounds.minX + this.bounds.maxX) / 2,
      y: this.bounds.minZ + 0.42,
      class: 'room-label',
      'text-anchor': 'middle',
    });
    label.textContent = `${room.name || 'Room'} · ${room.area_sqm ?? '—'} m²`;
    label.dataset.pxFont = 13;
    group.append(label);
    return group;
  }

  _walls(room) {
    const group = svg('g', { class: 'layer-walls' });
    for (const wall of room.walls || []) {
      const line = svg('line', {
        x1: wall.start.x, y1: wall.start.z, x2: wall.end.x, y2: wall.end.z,
        class: 'wall', 'stroke-linecap': 'square',
        'data-id': wall.wall_id,
      });
      line.dataset.pxStroke = 5;
      group.append(line);

      // A wider transparent line gives the thin wall a comfortable hit area —
      // a 5px stroke is painful to hit with a mouse and impossible on touch.
      const hit = svg('line', {
        x1: wall.start.x, y1: wall.start.z, x2: wall.end.x, y2: wall.end.z,
        class: 'hit', tabindex: '0', role: 'button',
        'aria-label': `Wall ${wall.wall_id}, ${wall.length_m} metres`,
      });
      hit.dataset.pxStroke = 16;
      const select = () => this.select({ kind: 'wall', id: wall.wall_id, data: wall });
      hit.addEventListener('click', select);
      hit.addEventListener('keydown', (event) => {
        if (event.key === 'Enter' || event.key === ' ') {
          event.preventDefault();
          select();
        }
      });
      group.append(hit);
    }
    return group;
  }

  _openings(room) {
    const group = svg('g', { class: 'layer-openings' });
    for (const [index, opening] of (room.openings || []).entries()) {
      const { start, end } = opening;
      const dx = end.x - start.x;
      const dz = end.z - start.z;
      const length = Math.hypot(dx, dz) || 1;
      const ux = dx / length;
      const uz = dz / length;

      // Clear the wall under the opening, then draw the symbol over it.
      const gap = svg('line', {
        x1: start.x, y1: start.z, x2: end.x, y2: end.z, class: 'opening-gap',
      });
      gap.dataset.pxStroke = 6;
      group.append(gap);

      if (opening.type === 'window') {
        const sill = svg('line', {
          x1: start.x, y1: start.z, x2: end.x, y2: end.z, class: 'window-pane',
        });
        sill.dataset.pxStroke = 2.4;
        group.append(sill);
      } else {
        const leaf = svg('line', {
          x1: start.x, y1: start.z,
          x2: start.x - uz * length, y2: start.z + ux * length,
          class: 'door-leaf',
        });
        leaf.dataset.pxStroke = 2;
        group.append(leaf);
        // Swing arc, drawn on the side the door opens into.
        const arc = svg('path', {
          d: `M ${end.x} ${end.z} A ${length} ${length} 0 0 1 ${start.x - uz * length} ${start.z + ux * length}`,
          class: 'door-arc',
          fill: 'none',
        });
        arc.dataset.pxStroke = 1.2;
        group.append(arc);
      }

      const hit = svg('line', {
        x1: start.x, y1: start.z, x2: end.x, y2: end.z,
        class: 'hit', tabindex: '0', role: 'button',
        'aria-label': `${opening.type}, ${opening.width_m} metres wide, on the ${opening.wall} wall`,
      });
      hit.dataset.pxStroke = 16;
      const select = () => this.select({ kind: 'opening', id: `opening-${index}`, data: opening });
      hit.addEventListener('click', select);
      hit.addEventListener('keydown', (event) => {
        if (event.key === 'Enter' || event.key === ' ') {
          event.preventDefault();
          select();
        }
      });
      group.append(hit);
    }
    return group;
  }

  _furniture(room) {
    const group = svg('g', { class: 'layer-furniture' });
    for (const [index, item] of (room.furniture_footprints || []).entries()) {
      const b = item.bbox;
      const rect = svg('rect', {
        x: b.center_x - b.width_m / 2,
        y: b.center_z - b.depth_m / 2,
        width: b.width_m,
        height: b.depth_m,
        rx: 0.05,
        class: 'furniture',
        tabindex: '0',
        role: 'button',
        'aria-label': `${item.type}, ${b.width_m} by ${b.depth_m} metres`,
      });
      rect.dataset.pxStroke = 1.5;
      const select = () => this.select({ kind: 'furniture', id: `furniture-${index}`, data: item });
      rect.addEventListener('click', select);
      rect.addEventListener('keydown', (event) => {
        if (event.key === 'Enter' || event.key === ' ') {
          event.preventDefault();
          select();
        }
      });
      group.append(rect);

      const label = svg('text', {
        x: b.center_x, y: b.center_z, class: 'furniture-label', 'text-anchor': 'middle',
        'dominant-baseline': 'middle',
      });
      label.textContent = item.type.split('/')[0].trim();
      label.dataset.pxFont = 9;
      group.append(label);
    }
    return group;
  }

  _dimensions(room) {
    const group = svg('g', { class: 'layer-dimensions', 'aria-hidden': 'true' });
    const offset = 0.45;
    for (const wall of room.walls || []) {
      const { start, end } = wall;
      const dx = end.x - start.x;
      const dz = end.z - start.z;
      const length = Math.hypot(dx, dz);
      if (length < 0.3) continue;
      const ux = dx / length;
      const uz = dz / length;
      // Outward normal: away from the room centre.
      const midX = (start.x + end.x) / 2;
      const midZ = (start.z + end.z) / 2;
      const cx = (this.bounds.minX + this.bounds.maxX) / 2;
      const cz = (this.bounds.minZ + this.bounds.maxZ) / 2;
      let nx = -uz;
      let nz = ux;
      if ((midX - cx) * nx + (midZ - cz) * nz < 0) {
        nx = -nx;
        nz = -nz;
      }

      const ax = start.x + nx * offset;
      const az = start.z + nz * offset;
      const bx = end.x + nx * offset;
      const bz = end.z + nz * offset;

      const line = svg('line', { x1: ax, y1: az, x2: bx, y2: bz, class: 'dim-line' });
      line.dataset.pxStroke = 0.9;
      group.append(line);
      for (const [tx, tz] of [[ax, az], [bx, bz]]) {
        const tick = svg('line', {
          x1: tx - nx * 0.1, y1: tz - nz * 0.1, x2: tx + nx * 0.1, y2: tz + nz * 0.1, class: 'dim-line',
        });
        tick.dataset.pxStroke = 0.9;
        group.append(tick);
      }

      const label = svg('text', {
        x: (ax + bx) / 2 + nx * 0.22,
        y: (az + bz) / 2 + nz * 0.22,
        class: 'dim-label', 'text-anchor': 'middle', 'dominant-baseline': 'middle',
      });
      label.textContent = `${length.toFixed(2)} m`;
      label.dataset.pxFont = 10;
      group.append(label);
    }
    return group;
  }

  _cameras() {
    const group = svg('g', { class: 'layer-cameras', 'aria-hidden': 'true' });
    const views = this.floorplan?.scan?.camera_views || [];
    for (const view of views) {
      const [x, , z] = view.position;
      const dot = svg('circle', { cx: x, cy: z, r: 0.08, class: 'camera-dot' });
      dot.dataset.pxStroke = 1;
      group.append(dot);
    }
    if (views.length > 1) {
      const path = views.map((v) => `${v.position[0]},${v.position[2]}`).join(' ');
      const poly = svg('polyline', { points: path, class: 'camera-path', fill: 'none' });
      poly.dataset.pxStroke = 1;
      group.append(poly);
    }
    return group;
  }

  /* ----------------------------------------------------------- selection */

  select(selection) {
    const same = this.selection && selection && this.selection.id === selection.id;
    this.selection = same ? null : selection;
    this._highlightSelection();
    this.onSelect(this.selection);
  }

  _highlightSelection() {
    this.root.querySelectorAll('.is-selected').forEach((n) => n.classList.remove('is-selected'));
    if (!this.selection) return;
    const { kind, id, data } = this.selection;
    if (kind === 'wall') {
      this.root.querySelector(`.wall[data-id="${id}"]`)?.classList.add('is-selected');
    } else if (kind === 'furniture') {
      const index = Number(id.split('-')[1]);
      this.root.querySelectorAll('.furniture')[index]?.classList.add('is-selected');
    } else if (kind === 'opening') {
      const selector = data.type === 'window' ? '.window-pane' : '.door-leaf';
      const index = Number(id.split('-')[1]);
      this.root.querySelectorAll(selector)[index]?.classList.add('is-selected');
    }
  }

  /* --------------------------------------------------------- interaction */

  _clientToPlan(event) {
    const point = this.svg.createSVGPoint();
    point.x = event.clientX;
    point.y = event.clientY;
    const ctm = this.svg.getScreenCTM();
    if (!ctm) return { x: 0, y: 0 };
    return point.matrixTransform(ctm.inverse());
  }

  _bindInteraction() {
    let dragging = false;
    let last = null;

    this.svg.addEventListener('pointerdown', (event) => {
      if (event.button !== 0) return;
      dragging = true;
      last = { x: event.clientX, y: event.clientY };
      this.svg.setPointerCapture(event.pointerId);
      this.svg.classList.add('is-panning');
    });

    this.svg.addEventListener('pointermove', (event) => {
      if (!dragging || !last) return;
      const perPx = this.view.w / (this.container.getBoundingClientRect().width || 1);
      this.view.x -= (event.clientX - last.x) * perPx;
      this.view.y -= (event.clientY - last.y) * perPx;
      last = { x: event.clientX, y: event.clientY };
      this._applyView();
    });

    const endDrag = (event) => {
      if (!dragging) return;
      dragging = false;
      last = null;
      this.svg.releasePointerCapture?.(event.pointerId);
      this.svg.classList.remove('is-panning');
    };
    this.svg.addEventListener('pointerup', endDrag);
    this.svg.addEventListener('pointercancel', endDrag);

    this.svg.addEventListener('wheel', (event) => {
      event.preventDefault();
      const point = this._clientToPlan(event);
      this._zoomAt(point.x, point.y, event.deltaY < 0 ? 1.15 : 1 / 1.15);
    }, { passive: false });

    // Clicking empty canvas clears the selection.
    this.svg.addEventListener('click', (event) => {
      if (event.target === this.svg || event.target.classList.contains('floor-fill')) {
        if (this.selection) {
          this.selection = null;
          this._highlightSelection();
          this.onSelect(null);
        }
      }
    });
  }
}
