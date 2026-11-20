# Features

Everything both services in this repository can do. They share no code and
deploy independently.

| | [Spatial Scanner](#spatial-scanner) | [StructuralSync](#structuralsync) |
|---|---|---|
| **Input** | A room scan: LiDAR point cloud + camera frames | A single kitchen photo |
| **Output** | 2D CAD floor design + explorable 3D room model | The same photo with new finishes |
| **Hardware** | CPU only | NVIDIA GPU (8 GB min, 16–24 GB comfortable) |
| **Runtime** | ~1.1 s per scan | ~25 s analysis†, 1–5 s per render |
| **Models** | None — deterministic geometry | Grounding DINO, SAM 2.1, MoGe-2, SigLIP 2, SDXL |
| **Tests** | 75 | 118 |
| **Port** | 8765 | 8000 |

† Analysis was measured at ~25 s on an M5 before the prompt vocabulary grew to cover rooms other
than kitchens, which took Grounding DINO from 4 passes to 7. The figure has not been re-measured
since and should be treated as a floor, not a current number.

---

# Spatial Scanner

*Camera + LiDAR → metric floor design and a 3D world you can walk through.*
Source: `services/scanner/`

## 1. Capture

### Clients
- **iOS client** (`capture/ios/`) — ARKit + LiDAR capture session, subsamples
  keyframes, writes RGB frames with camera poses and intrinsics
- **Custom rig client** (`capture/custom-rig/`) — USB camera + solid-state LiDAR
  node with timestamp sync and a static camera↔LiDAR extrinsic
- **Synthetic generator** (`capture/sample_generator.py`) — builds a complete
  scan bundle for a room of known dimensions: point cloud with sensor noise, a
  circular camera trajectory, poses and rendered frames

### Point-cloud support
- PLY in **ASCII**, **binary little-endian** and **binary big-endian**
- Colour channels as integers (0–255) or floats (0–1), under any of the common
  property names (`red`/`r`/`diffuse_red`)
- Clouds without colour get a neutral default so downstream code is uniform
- Truncated files are read up to the last complete vertex rather than rejected
- Writes ASCII or binary; binary is ~2× smaller and much faster to re-read
- Header-only validation, so an upload can be checked without parsing it fully

## 2. Reconstruction engine

### Ingest
- Scan-bundle validation **before** any work begins, with errors worth showing a user
- Pose loading that tolerates partial logs (a frame with no position is skipped, not fatal)
- Camera viewpoints transformed into the aligned coordinate frame

### Geometry
- **RANSAC floor detection with a gravity prior** — candidate planes must be
  near-horizontal, so a large wall can't be mistaken for the floor
- **Floor-vs-ceiling disambiguation** — if most of the cloud sits *below* the
  detected plane it's a ceiling, and the fit is retried without it
- Levelling to Y-up: floor normal → `[0,1,0]`, floor at `Y=0`, X/Z near the origin
- Statistical outlier removal (neighbour-distance based, k clamped to cloud size)
- Optional least-squares plane refit — available, off by default, with the
  measured trade-off documented in the code
- **Manhattan-world wall vectorization** via occupancy grid → contour → corner
  snapping, with a concave path for L-shaped rooms
- **Door and window detection** with width, height, sill height and swing
- **Furniture footprint detection** with bounding boxes and heights
- Memory-bounded grids — a mis-scaled cloud coarsens the resolution instead of
  exhausting RAM
- Metrics: floor area (m²/ft²), perimeter, per-wall lengths (m/ft)

### Export — 8 artifacts per scan
| 2D CAD | 3D model | Data |
|---|---|---|
| `floorplan.svg` — vector plan | `room_model.glb` — walkable | `floorplan.json` — full schema |
| `floorplan.dxf` — AutoCAD | `room_model.obj` — mesh | `colliders.json` — physics |
| `floorplan.pdf` — blueprint | | `aligned_point_cloud.ply` |

Plus `3dgs/transforms.json` — a NeRF/3DGS multi-view dataset for an external
Gaussian-splatting trainer.

Drawings include dimension lines, door swing arcs, window symbols, furniture
footprints and a room label. The 3D shell extrudes walls with **real openings
cut into them** (apron below a window, lintel above), glass panes for windows,
and furniture proxies.

### Pipeline
- Runs identically as a **library**, a **CLI** or through the **API** — same
  function, same arguments; only the progress callback differs
- Six reported stages with progress messages
- Deterministic: the same scan always produces the same measurements
- Configurable ceiling height; optional viewer sync

## 3. HTTP service

### Scan lifecycle
- A scan is a **resource**: `queued → processing → ready | failed`
- **Background worker** — reconstruction never runs inside a request; you get
  `202` immediately and poll for progress
- **Per-scan storage** — one directory each, atomic JSON metadata sidecar,
  survives restarts; no database
- **Interrupted scans are re-queued** when the service restarts
- **Automatic pruning** of the oldest finished scans past a configurable cap

### Endpoints (16)
| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/config` | Public: limits, whether a token is required |
| `GET` | `/api/health` | Status and worker queue depth |
| `GET` | `/api/scans` | List, newest first, paginated |
| `POST` | `/api/scans` | Create from an upload |
| `POST` | `/api/scans/sample` | Create from the bundled demo scan |
| `GET` | `/api/scans/{id}` | Status, progress, metrics, artifact list |
| `POST` | `/api/scans/{id}/reprocess` | Re-run on the same input |
| `DELETE` | `/api/scans/{id}` | Delete scan and artifacts |
| `GET` | `/api/scans/{id}/floorplan` | The floorplan schema |
| `GET` | `/api/scans/{id}/colliders` | Physics colliders |
| `GET` | `/api/scans/{id}/artifacts/{name}` | Any of the 8 artifacts |
| `GET` | `/api/scans/{id}/frames/{file}` | A captured RGB frame |

Plus legacy `/api/scan/latest`, `/process`, `/upload` and artifact routes that
resolve to the most recent finished scan, so older clients keep working.

### Uploads
- Point cloud + optional `camera_poses.json` + optional frame images, multipart
- **Streaming size cap** — aborted mid-write, so a huge upload can't fill the disk
- **Content validation** — a PLY is checked by parsing its header, not by its
  filename; poses are structurally validated; frame names are reduced to a safe
  basename with an allowlisted extension
- A rejected upload leaves no partial scan behind

### Security
- Optional shared-secret token (bearer header *or* query parameter, so `<img>`
  and download links work), compared in constant time
- Same-origin CORS by default; cross-origin is opt-in per origin
- **Artifact allowlist** — names are matched against a fixed set, never used as
  a path
- Frame serving resolves and confirms the path stays inside the scan directory
- Structured errors (`{error, detail}`) that never leak exception text, stack
  traces or absolute paths
- `run.sh --public` refuses to start without a token

### Operations
- OpenAPI docs at `/docs`
- 8 environment variables: `SCANNER_TOKEN`, `SCANNER_DATA`, `SCANNER_SAMPLE`,
  `SCANNER_VIEWER`, `SCANNER_CORS_ORIGINS`, `SCANNER_MAX_UPLOAD_MB`,
  `SCANNER_MAX_SCANS`, `SCANNER_LOG`

## 4. Web viewer

Plain ES modules, no bundler, Three.js vendored locally — works offline, no CDN
at runtime.

### Scan management
- Scan library with live status, progress bars and timings
- **Drag-and-drop upload** anywhere on the page, or a file picker, with upload progress
- One-click sample scan, reprocess, delete
- Automatic polling while anything is in flight, with completion/failure toasts

### 2D floor design
- **Live interactive SVG** built from the data (not a static image)
- Pan, zoom to cursor, fit-to-view; metres stay square at any zoom
- **Click any wall, opening or furniture item** to inspect its measurements
- Door swing arcs, window panes, dimension lines with ticks, furniture labels
- **5 toggleable layers** — dimensions, openings, furniture, camera path, grid
- Strokes and labels stay a constant on-screen size as you zoom

### 3D world
- **Dollhouse** — orbit the room from outside with damped controls
- **Walkthrough** — WASD/arrows, drag to look, Shift to run, 1.6 m eye height
- **Real collision** from the engine's `colliders.json`: walk *through a
  doorway*, but not through a wall or a sofa; wall-sliding; spawns at a
  guaranteed free position
- **Measurement** — click two surfaces for a distance in m and ft, with a
  running history you can delete individually
- **Camera pins** — click a pin to view the photo taken from that spot
- Wireframe toggle, recentre, **Save PNG** of the current view
- Shadows, grid, toggleable pins and ground grid

### Interface
- Export menu for all 8 artifacts
- Header metrics: floor area, perimeter, doors/windows, furniture, LiDAR points
- Keyboard shortcuts — `1` dollhouse, `2` walkthrough, `3` measure, `F` fit,
  `Esc` close library
- Responsive to phone width with three pane modes (2D / split / 3D)
- Accessible: keyboard-operable plan elements, visible focus rings, `aria-live`
  status, labelled controls, `prefers-reduced-motion`
- Loading skeletons, retryable error panels, toasts
- Light and dark themes

## 5. Quality
- **75 tests**: PLY round-trips across formats and endiannesses, malformed-file
  rejection, alignment and vectorization guards, ground-truth accuracy, the full
  HTTP API including auth, upload validation and path-traversal attempts
- Ruff-clean, pinned dependencies, isolated per-test data directories

### Measured accuracy (bundled sample, against known ground truth)
| Quantity | Truth | Measured | Error |
|---|---|---|---|
| Wall lengths | 6.00 m, 4.50 m | 6.00 m, 4.50 m | 0.00 % |
| Floor area | 27.00 m² | 26.97 m² | 0.11 % |
| Door width | 0.90 m | 0.90 m | 0.00 % |
| Window width | 2.00 m | 2.00 m | 0.00 % |

---

# StructuralSync

*One kitchen photo + a material catalogue → a photoreal finish swap.*
Source: `services/structsync/` · requires an NVIDIA GPU

## 1. Scene understanding
- **Surface detection** across 6 classes — countertop, cabinet, backsplash,
  floor, wall, ceiling — using Grounding DINO + SAM 2.1
- **Any indoor space, not only kitchens** — the occluder vocabulary covers beds,
  sofas, wardrobes, curtains, mirrors, rugs and bathroom fixtures, because what
  decides whether a bedroom photo comes out usable is not the surface list but
  whether the furniture in front of the wall is protected from the new paint.
  Added phrases cost one Grounding DINO pass per dozen, so analysis is slower
  than the kitchen-only version (4 passes → 7)
- **Ceiling orientation is signed** — a floor and a ceiling are the two largest
  horizontal planes in a room, so the filter tests which way a surface faces,
  not just that it is horizontal
- **Metric geometry** from a single photo via MoGe-2 (monocular depth + point map)
- **Surface model** — orientation filtering, RANSAC plane fitting, metric UVs,
  real-world area, and a metric scale prior taken from whichever standard
  real-world height the photo shows: counter-to-floor (0.91 m, tightly fixed) or
  ceiling-to-floor (2.5 m, looser), combined by weight when both are visible and
  rejected outright when the measurement is implausible
- **Cabinet detection** runs as its own pass with solidity and overlap filters
  tuned to avoid speckled false positives
- **Tap-to-correct** — click the overlay to add or remove regions when detection
  misses something; corrections persist and re-key the cache. A correction can
  target one region, or mint a new one where nothing was detected at all
- **Per-region addressing** — surfaces are instances (`wall#1`), not one mask per
  class, so two walls or two cabinet doors can take different materials. Walls
  split by plane, counters and backsplashes by connected run, cabinets per
  detected door; the floor never splits. Ids are assigned once per analysis
  (largest first) and survive corrections and restarts, so stored assignments and
  cache keys keep resolving

## 2. Rendering
- **Deterministic texture projection** — no generative model in the default path
- **Shading transfer** — large-scale light (masked low-pass) × fine darkening
  detail (grooves, bevels, contact shadows), auto-attenuated when the original
  surface was patterned so old patterns don't leak through
- **Edge snapping** via guided filter onto real image edges, hard composite, no halo
- **Countertop front edges** derived from geometry (3 cm drop along gravity,
  stops at occluders), textured with the top's SKU
- **Mirror reflections** (optional) — screen-space ray march through the point
  map on glossy horizontal surfaces, Schlick Fresnel, roughness-blurred
- **Grain matching** — counters and floors follow the surrounding surfaces;
  trilinear mips
- **Seam harmonisation** (optional) — SDXL-Inpainting + depth ControlNet,
  **gated**: rejected automatically if colour drift exceeds a measured
  threshold, falling back to the deterministic projection
- Render options: `edges`, `reflections`, `grain`, `material_maps`,
  `local_tint`, `harmonize`, `strength`
- Full-resolution output (≤2048 px) with masks snapped onto the original photo

## 3. Verification — every render is measured
- **`outside_change`** — mean pixel change outside the edited surfaces (should be ≈0)
- **`boundary_edge_recall`** — share of structural edges surviving at surface boundaries
- **drift ΔE** — median colour difference between harmonised and projected output
- **SKU recognition** — each edited surface is cropped, embedded with SigLIP 2
  and ranked against same-category SKUs; a pass means the assigned SKU ranks #1.
  This is independent of the renderer.
- **Bill of materials** — one line per material per class, with the areas of the
  regions it covers summed and those regions named

## 4. Catalogue
- **50 demo SKUs** across 6 categories (16 countertop, 11 cabinet, 7 backsplash,
  9 floor, 4 wall, 3 ceiling) with CC0 textures
- Per-SKU albedo plus optional normal and roughness maps; real-world tile size;
  gloss; vendor; price per m²
- **Semantic search** — describe a material in words ("dark marble") and SigLIP 2
  finds it; embeddings rebuild automatically
- Full CRUD with texture upload and replacement
- **Bulk import/export as one ZIP** — `catalog.json` *or* `catalog.csv`, merge or
  replace modes, all-or-nothing validation with row-numbered errors, a dry-run
  preview before you commit, and **crash-safe apply** (an interrupted import is
  finished or rolled back at the next start). Zip-slip is structurally
  impossible — member names are never used as write paths.

## 5. Web app
- **Projects** — upload a photo, surfaces detected in the background, pick
  materials or describe one, render
- **One material row per region**, each with its own overlay colour; a bare class
  name still assigns every region of that class
- **Before/after slider**, bill of materials, verification report per render
- **Render history** per project — revisit, download, delete
- **Catalogue manager** — add/edit/delete materials, upload textures, import/export
- Hash-based routing, no build step, no accounts, no database

## 6. Performance & operations
- **Two-tier result cache** — the same photo is analysed once (across all
  projects) and the same photo + corrections + materials + options is rendered
  once. Keys include pipeline version, code version, each SKU's catalogue entry
  *and* its texture pixels, so price or texture edits invalidate automatically.
- **Single-worker GPU executor** — requests queue for the one GPU; an LRU keeps
  recent scenes in memory to avoid re-decoding
- Crash recovery for both catalogue imports and interrupted analyses
- Optional shared-secret token, configurable data/catalogue paths, cache cap,
  preloading, log level (10 environment variables)
- 28 API endpoints; OpenAPI docs at `/docs`
- **118 tests** that never load a model (constructors are patched to raise),
  covering catalogue import/export round-trips, crash-mid-apply recovery,
  renderer determinism, edited-pixel isolation, geometry and the whole web API
- Region instancing is tested against the *silent* failure it would cause: the
  suite pins gravity recovery, the counter-height scale prior, texture-axis
  alignment, per-region slab edges and SKU-recognition category, each of which
  previously keyed on a bare class name and would skip itself — rather than
  fail — if handed a region id. Every one of those guards was checked by
  re-introducing the bug and confirming the test fails

---

# Platform

Shared across both services.

- **Monorepo** — `services/scanner` and `services/structsync`, independently
  testable and deployable
- **Containers** — scanner Dockerfile (~1 GB, non-root, healthcheck, data
  volume) and a compose file covering both, with StructuralSync behind a `gpu`
  profile so the default stack is CPU-only
- **Configuration** — environment variables throughout, `.env.example` for each
- **Named volumes** for all user data, so rebuilding images is safe
- **Documentation** — [architecture](architecture.md), [HTTP API](api.md),
  [deployment](../deploy/README.md), and a README per service
- **Tooling** — `make setup / run / test / lint / clean` at the repo root,
  delegating into the service

---

# Known limits

Stated plainly, because both services have real boundaries.

**Spatial Scanner**
- One room per scan — no doorway-based segmentation, no multi-floor
- Manhattan-world assumption; curved or non-orthogonal walls won't come out right
- Opening detection is heuristic, tuned on the sample room — sliding doors, bay
  windows and low sills are likely misclassified; door swing is assumed
- Furniture "classification" is bounding-box size, not recognition
- Ceiling height is assumed (2.8 m default), not measured
- The iOS client captures frames and poses but does not yet persist the depth
  buffer, so it cannot currently produce the bundle the engine consumes
- `export/splatting.py` prepares a 3DGS dataset; it does not train one
- The rig's camera↔LiDAR extrinsic is a hard-coded guess, not a calibration

**StructuralSync**
- Segmentation misses some regions; fix them with tap-to-correct
- No floor in view → no height prior at all, and areas drift ±20 %. A room with
  a floor but no counter now falls back to the ceiling pair, which is looser
- Room types other than kitchens are untested on photographs: the detector
  thresholds were tuned on 5 kitchen images, and the new occluder vocabulary has
  not been validated against a bedroom or bathroom sample set
- Very strong reflections on glossy originals are clamped, not removed
- Cabinet-style changes, furniture removal and adding geometry are out of scope
- Demo catalogue textures are procedural; real supplier scans remain the biggest
  realism gain
- The harmoniser needs significant VRAM and falls back to the projection when it
  cannot load
