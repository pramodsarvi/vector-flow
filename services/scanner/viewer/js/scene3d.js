/**
 * 3D room view: dollhouse orbit, first-person walkthrough, and measurement.
 *
 * Geometry comes from floorplan.json (walls extruded with real door/window
 * cut-outs, furniture proxies, camera pins). Walkthrough collision comes from
 * colliders.json — the physics representation the engine already exported and
 * the old viewer ignored in favour of clamping to the room's bounding box,
 * which let you walk through walls and furniture.
 */

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const WALL_THICKNESS = 0.15;
const EYE_HEIGHT = 1.6;
const WALK_SPEED = 2.6;      // m/s
const BODY_RADIUS = 0.32;    // collision capsule radius
const MODE = { DOLLHOUSE: 'dollhouse', WALK: 'walk', MEASURE: 'measure' };

export { MODE };

export class Scene3D {
  /**
   * @param {HTMLCanvasElement} canvas
   * @param {object} handlers onMeasure, onPinClick, onModeChange
   */
  constructor(canvas, handlers = {}) {
    this.canvas = canvas;
    this.handlers = handlers;
    this.mode = MODE.DOLLHOUSE;
    this.wireframe = false;
    this.disposed = false;

    this.floorplan = null;
    this.colliders = { segments: [], boxes: [] };
    this.bounds = { minX: 0, maxX: 6, minZ: 0, maxZ: 4.5, cx: 3, cz: 2.25, height: 2.8 };

    this.roomGroup = null;
    this.pinGroup = null;
    this.measureGroup = null;
    this.pins = [];
    this.measurements = [];
    this.pendingPoint = null;

    this.keys = new Set();
    this.look = { yaw: 0, pitch: 0 };
    this.clock = new THREE.Clock();
    this.raycaster = new THREE.Raycaster();

    this._initRenderer();
    this._initScene();
    this._bindEvents();
    this._animate();
  }

  /* ------------------------------------------------------------- set-up */

  _initRenderer() {
    // preserveDrawingBuffer keeps the last frame readable after compositing,
    // which is what makes snapshot() (and any screenshot tooling) reliable.
    this.renderer = new THREE.WebGLRenderer({
      canvas: this.canvas, antialias: true, preserveDrawingBuffer: true,
    });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  }

  _initScene() {
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x0b1020);
    this.scene.fog = new THREE.Fog(0x0b1020, 24, 60);

    this.camera = new THREE.PerspectiveCamera(55, 1, 0.05, 200);
    this.camera.position.set(8, 7, 10);

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.maxPolarAngle = Math.PI / 2 + 0.04;
    this.controls.target.set(3, 1.3, 2.2);

    this.scene.add(new THREE.AmbientLight(0xffffff, 0.68));

    const sun = new THREE.DirectionalLight(0xfff4e6, 1.05);
    sun.position.set(7, 12, 6);
    sun.castShadow = true;
    sun.shadow.mapSize.set(1024, 1024);
    sun.shadow.camera.near = 1;
    sun.shadow.camera.far = 50;
    const extent = 14;
    Object.assign(sun.shadow.camera, { left: -extent, right: extent, top: extent, bottom: -extent });
    this.scene.add(sun);

    const fill = new THREE.DirectionalLight(0xaac4e8, 0.38);
    fill.position.set(-6, 5, -5);
    this.scene.add(fill);

    this.grid = new THREE.GridHelper(40, 80, 0x2d4a7c, 0x16213c);
    this.grid.position.y = -0.012;
    this.scene.add(this.grid);

    this._resize();
  }

  _bindEvents() {
    this.resizeObserver = new ResizeObserver(() => this._resize());
    this.resizeObserver.observe(this.canvas.parentElement || this.canvas);

    this._onKeyDown = (event) => {
      // Never swallow keys meant for a text field or a dialog.
      const target = event.target;
      if (target instanceof HTMLElement && /input|textarea|select/i.test(target.tagName)) return;
      this.keys.add(event.code);
      if (this.mode === MODE.WALK && WALK_KEYS.has(event.code)) event.preventDefault();
    };
    this._onKeyUp = (event) => this.keys.delete(event.code);
    window.addEventListener('keydown', this._onKeyDown);
    window.addEventListener('keyup', this._onKeyUp);

    this._onPointerDown = (event) => this._handlePointerDown(event);
    this._onPointerMove = (event) => this._handlePointerMove(event);
    this._onPointerUp = () => { this.dragging = false; };
    this.canvas.addEventListener('pointerdown', this._onPointerDown);
    this.canvas.addEventListener('pointermove', this._onPointerMove);
    window.addEventListener('pointerup', this._onPointerUp);
    this.canvas.addEventListener('click', (event) => this._handleClick(event));
  }

  destroy() {
    this.disposed = true;
    cancelAnimationFrame(this._frame);
    this.resizeObserver?.disconnect();
    window.removeEventListener('keydown', this._onKeyDown);
    window.removeEventListener('keyup', this._onKeyUp);
    window.removeEventListener('pointerup', this._onPointerUp);
    this.canvas.removeEventListener('pointerdown', this._onPointerDown);
    this.canvas.removeEventListener('pointermove', this._onPointerMove);
    this.controls.dispose();
    disposeTree(this.scene);
    this.renderer.dispose();
  }

  _resize() {
    const parent = this.canvas.parentElement || this.canvas;
    const { width, height } = parent.getBoundingClientRect();
    if (!width || !height) return;
    this.renderer.setSize(width, height, false);
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
  }

  /* -------------------------------------------------------------- model */

  load(floorplan, colliders) {
    this.floorplan = floorplan;
    this.colliders = parseColliders(colliders);
    this._computeBounds();
    this.clearMeasurements();

    for (const group of [this.roomGroup, this.pinGroup]) {
      if (group) {
        this.scene.remove(group);
        disposeTree(group);
      }
    }
    this.roomGroup = this._buildRoom();
    this.scene.add(this.roomGroup);
    this.pinGroup = this._buildPins();
    this.scene.add(this.pinGroup);

    if (this.wireframe) this.setWireframe(true);
    this.setMode(MODE.DOLLHOUSE);
  }

  _computeBounds() {
    const polygon = this.floorplan?.room?.polygon || [];
    if (!polygon.length) return;
    const xs = polygon.map((p) => p.x);
    const zs = polygon.map((p) => p.z);
    this.bounds = {
      minX: Math.min(...xs), maxX: Math.max(...xs),
      minZ: Math.min(...zs), maxZ: Math.max(...zs),
      height: this.floorplan.ceiling_height_m || 2.8,
    };
    this.bounds.cx = (this.bounds.minX + this.bounds.maxX) / 2;
    this.bounds.cz = (this.bounds.minZ + this.bounds.maxZ) / 2;
  }

  _buildRoom() {
    const group = new THREE.Group();
    const room = this.floorplan.room;
    const height = this.bounds.height;

    // Floor & ceiling from the scanned polygon. THREE.Shape works in XY, so z
    // is negated and the mesh rotated flat — the standard plan-to-world flip.
    const shape = new THREE.Shape((room.polygon || []).map((p) => new THREE.Vector2(p.x, -p.z)));
    const slabGeo = new THREE.ShapeGeometry(shape);

    const floor = new THREE.Mesh(slabGeo, new THREE.MeshStandardMaterial({
      color: 0x8a5b33, roughness: 0.55, metalness: 0.04, side: THREE.DoubleSide,
    }));
    floor.rotation.x = -Math.PI / 2;
    floor.receiveShadow = true;
    floor.name = 'floor';
    group.add(floor);

    const ceiling = new THREE.Mesh(slabGeo.clone(), new THREE.MeshStandardMaterial({
      color: 0xffffff, transparent: true, opacity: 0.12, roughness: 0.9, side: THREE.DoubleSide,
    }));
    ceiling.rotation.x = -Math.PI / 2;
    ceiling.position.y = height;
    ceiling.name = 'ceiling';
    group.add(ceiling);

    const wallMat = new THREE.MeshStandardMaterial({ color: 0xe6ebf3, roughness: 0.88, side: THREE.DoubleSide });
    const glassMat = new THREE.MeshStandardMaterial({
      color: 0x55c4f5, transparent: true, opacity: 0.34, roughness: 0.08, metalness: 0.1,
    });

    for (const wall of room.walls || []) {
      const sx = wall.start.x;
      const sz = wall.start.z;
      const dx = wall.end.x - sx;
      const dz = wall.end.z - sz;
      const length = Math.hypot(dx, dz);
      if (length < 1e-3) continue;
      const ux = dx / length;
      const uz = dz / length;
      const rotY = -Math.atan2(dz, dx);

      const addPiece = (from, to, y0, y1, material = wallMat, thickness = WALL_THICKNESS) => {
        if (to - from < 0.01 || y1 - y0 < 0.01) return;
        const mesh = new THREE.Mesh(new THREE.BoxGeometry(to - from, y1 - y0, thickness), material);
        const mid = (from + to) / 2;
        mesh.position.set(sx + ux * mid, (y0 + y1) / 2, sz + uz * mid);
        mesh.rotation.y = rotY;
        mesh.castShadow = true;
        mesh.receiveShadow = true;
        mesh.name = 'wall';
        group.add(mesh);
      };

      let cursor = 0;
      for (const { opening, from, to } of openingsOnWall(wall, room.openings || [])) {
        addPiece(cursor, from, 0, height);
        const sill = opening.sill_height_m || 0;
        const top = Math.min(height, sill + (opening.height_m || 2.1));
        addPiece(from, to, 0, sill);     // apron below a window
        addPiece(from, to, top, height); // lintel above the opening
        if (opening.type === 'window') addPiece(from, to, sill, top, glassMat, 0.04);
        cursor = Math.max(cursor, to);
      }
      addPiece(cursor, length, 0, height);
    }

    for (const item of room.furniture_footprints || []) {
      const b = item.bbox;
      const h = b.height_m || 0.8;
      const mesh = new THREE.Mesh(
        new THREE.BoxGeometry(b.width_m, h, b.depth_m),
        new THREE.MeshStandardMaterial({ color: furnitureColor(item.type), roughness: 0.65 }),
      );
      mesh.position.set(b.center_x, h / 2, b.center_z);
      mesh.castShadow = true;
      mesh.receiveShadow = true;
      mesh.name = 'furniture';
      mesh.userData.label = item.type;
      group.add(mesh);
    }
    return group;
  }

  _buildPins() {
    const group = new THREE.Group();
    this.pins = [];
    const views = this.floorplan?.scan?.camera_views || [];
    if (!views.length) return group;

    const geometry = new THREE.ConeGeometry(0.085, 0.2, 10);
    const material = new THREE.MeshBasicMaterial({ color: 0xfacc15 });
    const path = [];

    for (const view of views) {
      const position = new THREE.Vector3(...view.position);
      path.push(position);
      const pin = new THREE.Mesh(geometry, material);
      pin.position.copy(position);
      pin.rotation.x = Math.PI;
      pin.name = 'camera-pin';
      pin.userData.view = view;
      this.pins.push(pin);
      group.add(pin);
    }

    if (path.length > 1) {
      const line = new THREE.Line(
        new THREE.BufferGeometry().setFromPoints([...path, path[0]]),
        new THREE.LineDashedMaterial({ color: 0xeab308, dashSize: 0.18, gapSize: 0.12 }),
      );
      line.computeLineDistances();
      group.add(line);
    }
    return group;
  }

  /* --------------------------------------------------------------- modes */

  setMode(mode) {
    this.mode = mode;
    this.controls.enabled = mode === MODE.DOLLHOUSE;
    this.canvas.style.cursor = mode === MODE.MEASURE ? 'crosshair' : mode === MODE.WALK ? 'grab' : 'default';
    if (mode === MODE.DOLLHOUSE) this.frameDollhouse();
    if (mode === MODE.WALK) this._enterWalk();
    if (mode !== MODE.MEASURE) this.pendingPoint = null;
    this.handlers.onModeChange?.(mode);
  }

  frameDollhouse() {
    const span = Math.max(this.bounds.maxX - this.bounds.minX, this.bounds.maxZ - this.bounds.minZ) || 6;
    this.camera.position.set(this.bounds.cx + span * 0.72, span * 1.05, this.bounds.cz + span * 1.15);
    this.controls.target.set(this.bounds.cx, this.bounds.height / 2, this.bounds.cz);
    this.controls.update();
  }

  _enterWalk() {
    const spawn = this._findSpawn();
    this.camera.position.set(spawn.x, EYE_HEIGHT, spawn.z);
    this.look.yaw = 0;
    this.look.pitch = 0;
    this._applyLook();
  }

  /**
   * A standing position that is not inside a wall or a piece of furniture.
   *
   * The room centre is the obvious choice and usually right, but it is also
   * exactly where a table tends to be — spawning there leaves the walker
   * stuck. Fall back to a spiral of candidates and take the first free one.
   */
  _findSpawn() {
    const { cx, cz, minX, maxX, minZ, maxZ } = this.bounds;
    if (!this._blocked(cx, cz)) return { x: cx, z: cz };

    const step = 0.25;
    const maxRings = Math.ceil(Math.max(maxX - minX, maxZ - minZ) / step);
    for (let ring = 1; ring <= maxRings; ring += 1) {
      const radius = ring * step;
      for (let a = 0; a < 16; a += 1) {
        const angle = (a / 16) * Math.PI * 2;
        const x = cx + Math.cos(angle) * radius;
        const z = cz + Math.sin(angle) * radius;
        if (x < minX || x > maxX || z < minZ || z > maxZ) continue;
        if (!this._blocked(x, z)) return { x, z };
      }
    }
    return { x: cx, z: cz }; // nowhere free: stand at the centre anyway
  }

  _applyLook() {
    this.look.pitch = THREE.MathUtils.clamp(this.look.pitch, -Math.PI / 2 + 0.05, Math.PI / 2 - 0.05);
    const euler = new THREE.Euler(this.look.pitch, this.look.yaw, 0, 'YXZ');
    this.camera.quaternion.setFromEuler(euler);
  }

  setWireframe(enabled) {
    this.wireframe = enabled;
    this.roomGroup?.traverse((node) => {
      if (node.isMesh && node.material && 'wireframe' in node.material) {
        node.material.wireframe = enabled;
      }
    });
  }

  setPinsVisible(visible) {
    if (this.pinGroup) this.pinGroup.visible = visible;
  }

  setGridVisible(visible) {
    this.grid.visible = visible;
  }

  /* ---------------------------------------------------------- interaction */

  _handlePointerDown(event) {
    if (this.mode !== MODE.WALK) return;
    this.dragging = true;
    this.lastPointer = { x: event.clientX, y: event.clientY };
  }

  _handlePointerMove(event) {
    if (this.mode !== MODE.WALK || !this.dragging) return;
    const dx = event.clientX - this.lastPointer.x;
    const dy = event.clientY - this.lastPointer.y;
    this.lastPointer = { x: event.clientX, y: event.clientY };
    this.look.yaw -= dx * 0.0032;
    this.look.pitch -= dy * 0.0032;
    this._applyLook();
  }

  _pointerToNdc(event) {
    const rect = this.canvas.getBoundingClientRect();
    return new THREE.Vector2(
      ((event.clientX - rect.left) / rect.width) * 2 - 1,
      -((event.clientY - rect.top) / rect.height) * 2 + 1,
    );
  }

  _handleClick(event) {
    this.raycaster.setFromCamera(this._pointerToNdc(event), this.camera);

    // Camera pins take priority — they are small and deliberately clicked.
    if (this.pinGroup?.visible && this.pins.length) {
      const hit = this.raycaster.intersectObjects(this.pins, false)[0];
      if (hit) {
        this.handlers.onPinClick?.(hit.object.userData.view);
        return;
      }
    }
    if (this.mode !== MODE.MEASURE || !this.roomGroup) return;

    const hit = this.raycaster.intersectObject(this.roomGroup, true)[0];
    if (!hit) return;
    this._addMeasurePoint(hit.point.clone());
  }

  /* --------------------------------------------------------- measurement */

  _addMeasurePoint(point) {
    if (!this.pendingPoint) {
      this.pendingPoint = point;
      this._drawMeasurements();
      this.handlers.onMeasure?.(this.measurements, { pending: true });
      return;
    }
    const from = this.pendingPoint;
    this.pendingPoint = null;
    const distance = from.distanceTo(point);
    this.measurements.push({
      id: `m${Date.now()}`,
      from: from.toArray().map((n) => Number(n.toFixed(3))),
      to: point.toArray().map((n) => Number(n.toFixed(3))),
      distance_m: Number(distance.toFixed(3)),
    });
    this._drawMeasurements();
    this.handlers.onMeasure?.(this.measurements, { pending: false });
  }

  removeMeasurement(id) {
    this.measurements = this.measurements.filter((m) => m.id !== id);
    this._drawMeasurements();
    this.handlers.onMeasure?.(this.measurements, { pending: false });
  }

  clearMeasurements() {
    this.measurements = [];
    this.pendingPoint = null;
    this._drawMeasurements();
    this.handlers.onMeasure?.(this.measurements, { pending: false });
  }

  _drawMeasurements() {
    if (this.measureGroup) {
      this.scene.remove(this.measureGroup);
      disposeTree(this.measureGroup);
    }
    this.measureGroup = new THREE.Group();

    const markerGeo = new THREE.SphereGeometry(0.045, 12, 12);
    const markerMat = new THREE.MeshBasicMaterial({ color: 0x34d399 });
    const lineMat = new THREE.LineBasicMaterial({ color: 0x34d399 });

    for (const measurement of this.measurements) {
      const a = new THREE.Vector3(...measurement.from);
      const b = new THREE.Vector3(...measurement.to);
      this.measureGroup.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints([a, b]), lineMat));
      for (const point of [a, b]) {
        const marker = new THREE.Mesh(markerGeo, markerMat);
        marker.position.copy(point);
        this.measureGroup.add(marker);
      }
    }
    if (this.pendingPoint) {
      const marker = new THREE.Mesh(markerGeo, new THREE.MeshBasicMaterial({ color: 0xfbbf24 }));
      marker.position.copy(this.pendingPoint);
      this.measureGroup.add(marker);
    }
    this.scene.add(this.measureGroup);
  }

  /* ----------------------------------------------------------- walk loop */

  _walk(delta) {
    let forward = 0;
    let strafe = 0;
    if (this.keys.has('KeyW') || this.keys.has('ArrowUp')) forward += 1;
    if (this.keys.has('KeyS') || this.keys.has('ArrowDown')) forward -= 1;
    if (this.keys.has('KeyD') || this.keys.has('ArrowRight')) strafe += 1;
    if (this.keys.has('KeyA') || this.keys.has('ArrowLeft')) strafe -= 1;
    if (!forward && !strafe) return;

    const speed = WALK_SPEED * (this.keys.has('ShiftLeft') || this.keys.has('ShiftRight') ? 2 : 1) * delta;
    const sin = Math.sin(this.look.yaw);
    const cos = Math.cos(this.look.yaw);
    // Forward in a Y-up, -Z-forward camera frame.
    let dx = (-sin * forward + cos * strafe) * speed;
    let dz = (-cos * forward - sin * strafe) * speed;

    const next = this._resolveCollision(
      this.camera.position.x, this.camera.position.z,
      this.camera.position.x + dx, this.camera.position.z + dz,
    );
    this.camera.position.x = next.x;
    this.camera.position.z = next.z;
    this.camera.position.y = EYE_HEIGHT;
  }

  /**
   * Push the walker out of walls and furniture.
   *
   * Each axis is resolved separately so sliding along a wall still works
   * instead of stopping dead. Wall segments come from the engine's collider
   * export, which already omits door gaps — so doorways are walkable and the
   * wall beside them is not.
   */
  _resolveCollision(fromX, fromZ, toX, toZ) {
    let x = toX;
    let z = toZ;

    for (const axis of ['x', 'z']) {
      const testX = axis === 'x' ? x : fromX;
      const testZ = axis === 'z' ? z : fromZ;
      if (this._blocked(testX, testZ)) {
        if (axis === 'x') x = fromX;
        else z = fromZ;
      }
    }
    // Both axes blocked individually but the diagonal was fine: stay put.
    if (this._blocked(x, z)) return { x: fromX, z: fromZ };
    return { x, z };
  }

  _blocked(x, z) {
    for (const segment of this.colliders.segments) {
      if (distanceToSegment(x, z, segment) < BODY_RADIUS) return true;
    }
    for (const box of this.colliders.boxes) {
      if (
        x > box.minX - BODY_RADIUS && x < box.maxX + BODY_RADIUS &&
        z > box.minZ - BODY_RADIUS && z < box.maxZ + BODY_RADIUS
      ) return true;
    }
    return false;
  }

  /* ---------------------------------------------------------- main loop */

  _animate() {
    this._frame = requestAnimationFrame(() => this._animate());
    if (this.disposed) return;
    const delta = Math.min(this.clock.getDelta(), 0.1);

    if (this.mode === MODE.WALK) this._walk(delta);
    else this.controls.update();

    this.renderer.render(this.scene, this.camera);
  }

  /** A data URL of the current frame, for the "save screenshot" action. */
  snapshot() {
    this.renderer.render(this.scene, this.camera);
    return this.renderer.domElement.toDataURL('image/png');
  }
}

const WALK_KEYS = new Set([
  'KeyW', 'KeyA', 'KeyS', 'KeyD', 'ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight', 'Space',
]);

/** Openings lying on a wall, as intervals in metres from the wall's start. */
function openingsOnWall(wall, openings) {
  const sx = wall.start.x;
  const sz = wall.start.z;
  const dx = wall.end.x - sx;
  const dz = wall.end.z - sz;
  const length = Math.hypot(dx, dz) || 1;
  const ux = dx / length;
  const uz = dz / length;
  const along = (p) => (p.x - sx) * ux + (p.z - sz) * uz;
  const offset = (p) => Math.abs(-(p.x - sx) * uz + (p.z - sz) * ux);

  return openings
    .filter((o) => offset(o.start) < 0.25 && offset(o.end) < 0.25)
    .map((opening) => ({
      opening,
      from: Math.max(0, Math.min(along(opening.start), along(opening.end))),
      to: Math.min(length, Math.max(along(opening.start), along(opening.end))),
    }))
    .filter((entry) => entry.to - entry.from > 0.05)
    .sort((a, b) => a.from - b.from);
}

/** Flatten colliders.json into 2D segments and AABBs for the walk loop. */
function parseColliders(colliders) {
  const segments = [];
  const boxes = [];
  for (const collider of colliders?.colliders || []) {
    if (collider.type === 'wall_segment') {
      const [x1, , z1] = collider.start;
      const [x2, , z2] = collider.end;
      segments.push({ x1, z1, x2, z2 });
    } else if (collider.type === 'box') {
      const [cx, , cz] = collider.center;
      const [w, , d] = collider.extents;
      boxes.push({ minX: cx - w / 2, maxX: cx + w / 2, minZ: cz - d / 2, maxZ: cz + d / 2 });
    }
  }
  return { segments, boxes };
}

function distanceToSegment(px, pz, { x1, z1, x2, z2 }) {
  const dx = x2 - x1;
  const dz = z2 - z1;
  const lengthSq = dx * dx + dz * dz;
  if (lengthSq < 1e-9) return Math.hypot(px - x1, pz - z1);
  let t = ((px - x1) * dx + (pz - z1) * dz) / lengthSq;
  t = Math.max(0, Math.min(1, t));
  return Math.hypot(px - (x1 + t * dx), pz - (z1 + t * dz));
}

function furnitureColor(type) {
  if (/sofa|seat|couch|chair/i.test(type)) return 0x2f5681;
  if (/table|desk/i.test(type)) return 0xa06a37;
  if (/bed/i.test(type)) return 0x9aa3b2;
  return 0x64748b;
}

function disposeTree(root) {
  root.traverse((node) => {
    node.geometry?.dispose();
    const material = node.material;
    if (Array.isArray(material)) material.forEach((m) => m.dispose());
    else material?.dispose();
  });
}
