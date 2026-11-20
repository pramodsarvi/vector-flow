# Spatial Scanner & Floor Designer

*A service in this repository — see the [repository README](../../README.md) for the others.*

Capture a room with **a camera and a LiDAR (time-of-flight) depth sensor**, and get back:

1. **A metric 2D floor design** — floor plane, perimeter walls, doors, windows and furniture
   footprints, exported as `.dxf`, `.svg`, `.pdf` and `.json`.
2. **An explorable 3D room model** — `.glb` / `.obj` with physics colliders, viewable as a
   dollhouse, walked through in first person, and measurable point-to-point.

Everything runs as one process: a FastAPI service that does the reconstruction and serves the
web viewer. No GPU, no database, no build step.

---

## Quick start

```bash
make setup      # create .venv and install pinned dependencies (uses uv if present)
make run        # start the service
```

Then open **<http://127.0.0.1:8765/viewer/>** and click **Sample scan**. Processing takes about
two seconds; the floor design and 3D model appear when it finishes.

API documentation is at <http://127.0.0.1:8765/docs>.

```bash
make test       # 75 tests, ~20 s, no GPU and no network required
make lint       # ruff
make scan       # run the pipeline from the command line instead of the API
                # (equivalently: python -m engine.pipeline, from the project root)
```

Requires Python 3.11+ (verified on 3.12). If you prefer not to use `make`:

```bash
uv venv -p 3.12 .venv
uv pip install -p .venv/bin/python -r requirements-dev.txt
./run.sh
```

---

## Configuration

Every setting is an environment variable; copy `.env.example` to `.env` to set them.

| Variable | Default | Meaning |
|---|---|---|
| `SCANNER_TOKEN` | *(empty)* | Require this token on every API call. **Set it before exposing the service beyond localhost.** |
| `SCANNER_DATA` | `./data` | Where scans are stored |
| `SCANNER_CORS_ORIGINS` | *(empty)* | Allowed browser origins; empty means same-origin only |
| `SCANNER_MAX_UPLOAD_MB` | `512` | Largest accepted upload |
| `SCANNER_MAX_SCANS` | `50` | Scans kept before the oldest finished ones are pruned |
| `SCANNER_LOG` | `INFO` | Log level |

`./run.sh --public` refuses to start without `SCANNER_TOKEN`, so the service cannot be exposed
to a network by accident.

---


---

## The viewer

Plain ES modules served straight from `viewer/` — no bundler, no CDN at runtime.

- **2D floor design** drawn from `floorplan.json` as live SVG: pan, zoom, hover and click walls,
  openings and furniture to inspect their measurements; layers for dimensions, openings,
  furniture, the camera path and the grid.
- **3D world** with three modes: dollhouse orbit, first-person walkthrough (WASD, drag to look,
  Shift to run) and point-to-point measurement with a running list of measurements.
  Walkthrough collision uses the engine's exported `colliders.json`, so you can walk through a
  doorway but not through a wall or a sofa.
- **Scan library** with upload (drag a `.ply` anywhere on the page), live progress, reprocess
  and delete.
- Keyboard shortcuts: `1` dollhouse, `2` walkthrough, `3` measure, `F` fit the plan.
- Responsive down to phone width, keyboard-operable, with visible focus and `aria-live` status.

---

---

## Accuracy on the bundled sample

The sample room is generated with known dimensions, so these are measured against ground truth,
not against previous output. `tests/test_pipeline.py` asserts every row.

| Quantity | Ground truth | Measured | Error |
|---|---|---|---|
| Wall lengths | 6.00 m, 4.50 m | 6.00 m, 4.50 m | 0.00 % |
| Floor area | 27.00 m² | 26.97 m² | 0.11 % |
| Door width | 0.90 m | 0.90 m | 0.00 % |
| Window width | 2.00 m | 2.00 m | 0.00 % |
| Window sill height | 0.90 m | 0.85 m | 0.05 m |
| Furniture detected | 2 | 2 | — |
| Camera viewpoints | 12 | 12 | — |

Openings are measured on a 0.10 m grid, which is the resolution limit on these figures.

---

---

## Known limitations

Worth knowing before pointing this at a real building:

- **One room per scan.** There is no doorway-based room segmentation or multi-floor support.
- **Manhattan-world assumption.** The vectorizer regularizes to axis-aligned walls, with a
  concave path for L-shaped rooms. Rooms with curved or non-orthogonal walls will not come out
  right.
- **Opening detection is heuristic.** Doors and windows are found by looking for gaps at eye
  level, with thresholds tuned on the sample room. Sliding doors, bay windows and low sills are
  likely to be misclassified. Door swing direction is assumed, not measured.
- **Furniture classification is bounding-box size, not recognition.** "Sofa" means "wider than
  1.6 m", nothing more.
- **Ceiling height is assumed** (2.8 m by default, `--ceiling-height` to change it); it is not
  measured from the cloud.
- **The iOS client does not yet write a point cloud.** `RoomScannerSession.swift` captures
  keyframes and poses but does not persist the depth buffer or RoomPlan geometry, so it cannot
  currently produce the bundle the engine consumes.
- **`export/splatting.py` prepares a 3DGS dataset; it does not train one.** It writes
  `transforms.json` for an external splatting trainer.
- **The rig extrinsics in `camera_lidar_sync.py` are a hard-coded guess**, not a calibration.

---

---

## Development

```bash
make test                   # the whole suite
.venv/bin/python -m pytest tests/test_pipeline.py -v    # one file
make lint
```

The suite covers PLY round-trips (ASCII, binary, both endiannesses, malformed files), the
alignment and vectorization guards, the pipeline's measurements against ground truth, and the
whole HTTP API including auth, upload validation and path-traversal attempts. Each test runs
against its own temporary data directory.

Dependencies are pinned in `requirements.txt` to the versions the pipeline was verified with.
Upgrade deliberately: bump a pin, run `make test`, and check the accuracy table still holds.

## Where things are

| I want to… | Look in |
|---|---|
| change how a bundle is read | `engine/ingest/` |
| change floor detection or wall extraction | `engine/geometry/` |
| add or change an output format | `engine/export/` |
| change an endpoint, auth or storage | `engine/service/` |
| change the 2D plan or 3D view | `viewer/js/` |
| understand the whole flow | [`docs/architecture.md`](../../docs/architecture.md) |
| look up an endpoint | [`docs/api.md`](../../docs/api.md) |
| deploy it | [`deploy/`](../../deploy/) |
