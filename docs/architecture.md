# Architecture

How a capture becomes a floor design and a 3D model, and where each piece of
that journey lives in the tree.

## How it works

```text
[Camera (RGB) + LiDAR (depth)]
          │
          ▼
 1. Ingest a scan bundle          raw_lidar_cloud.ply + camera_poses.json + images/
          │                        (PLY: ASCII or binary, little- or big-endian)
          ▼
 2. Floor detection & levelling   RANSAC with a gravity prior, then a floor-vs-ceiling
          │                        check so a dense ceiling cannot be mistaken for the floor
          ▼
 3. Wall slicing                  points in Y ∈ [0.15 m, 1.20 m]
          │
          ▼
 4. Vectorization                 occupancy grid → contour → Manhattan-snapped polygon,
          │                        plus door/window and furniture detection
          ▼
 5. Export                        2D: floorplan.{json,svg,dxf,pdf}
          │                       3D: room_model.{obj,glb}, colliders.json
          │                       3DGS: 3dgs/transforms.json
          ▼
 6. Web viewer                    interactive 2D plan + 3D dollhouse / walkthrough / measure
```

### Layout

```text
spatial-scanner-product/
├── capture/                             how a scan is produced
│   ├── ios/RoomScannerSession.swift     ARKit + LiDAR capture session
│   ├── custom-rig/camera_lidar_sync.py  USB camera + solid-state LiDAR sync node
│   ├── sample_generator.py              synthetic scan generator
│   └── sample-data/                     the bundled demo scan
│
├── engine/                              how a scan becomes deliverables
│   ├── ingest/                          reading capture data
│   │   ├── ply_io.py                    point-cloud IO (ASCII + binary PLY)
│   │   └── bundle.py                    bundle validation, poses, camera views
│   ├── geometry/                        making sense of the cloud
│   │   ├── alignment.py                 floor detection & levelling
│   │   └── vectorize.py                 walls, openings, furniture, schema
│   ├── export/                          writing deliverables
│   │   ├── cad.py                       SVG / DXF / blueprint PDF
│   │   ├── mesh.py                      OBJ / GLB / physics colliders
│   │   └── splatting.py                 NeRF/3DGS transforms.json
│   ├── service/                         serving it over HTTP
│   │   ├── app.py                       routes
│   │   ├── store.py                     scan storage (one directory per scan)
│   │   ├── jobs.py                      background worker queue
│   │   ├── uploads.py                   upload validation
│   │   ├── schemas.py                   response models
│   │   └── settings.py                  configuration
│   └── pipeline.py                      the orchestrator: ingest → geometry → export
│
├── viewer/                              how a scan is explored
│   ├── index.html · styles.css
│   ├── js/{app,api,ui,plan2d,scene3d}.js
│   └── vendor/                          Three.js, vendored rather than CDN-loaded
│
└── tests/                               pytest suite
```

Each engine package is one stage of the same journey, and the dependency
direction only ever runs forwards: `ingest → geometry → export`, with `service`
and `pipeline` orchestrating on top. Nothing in `geometry` knows about HTTP, and
nothing in `export` reads a file off the capture bundle.

---


---

## Why it is split this way

The engine's four packages are the four things that actually happen to a scan,
in order, and the dependency direction only ever runs forwards:

```text
ingest  ->  geometry  ->  export
                 ^
            pipeline  (orchestrates the three)
                 ^
             service  (runs pipeline on a worker, serves the results)
```

- **`ingest`** is the only layer that knows what a capture bundle looks like on
  disk. Nothing downstream opens a file.
- **`geometry`** is pure computation on arrays. It knows nothing about HTTP,
  files or output formats, which is why it is the easiest part to test.
- **`export`** only consumes the floorplan schema. Adding a new output format
  means adding a module here and one call in the pipeline.
- **`service`** is the only layer that knows about HTTP, and it never does
  reconstruction itself — it hands work to a background worker.

A practical consequence: the pipeline runs identically from the CLI, from a
test, and from the API, because all three call the same function with the same
arguments. The only difference is what they pass as the progress callback.

---

## The scan lifecycle

A scan is a resource with a lifecycle. You create it, the background worker processes it, and
you poll until it is `ready`. Reconstruction never runs inside a request handler, so the API
stays responsive while a scan is being processed, and one worker runs at a time because the
pipeline seeds the global RNG for reproducible measurements.

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/config` | Public: limits and whether a token is required |
| `GET` | `/api/health` | Service status and worker queue depth |
| `GET` | `/api/scans` | List scans, newest first |
| `POST` | `/api/scans` | Create from an uploaded bundle (`cloud`, optional `poses`, `frames`) |
| `POST` | `/api/scans/sample` | Create from the bundled demo capture |
| `GET` | `/api/scans/{id}` | Status, progress, metrics, available artifacts |
| `POST` | `/api/scans/{id}/reprocess` | Re-run the pipeline on the same input |
| `DELETE` | `/api/scans/{id}` | Delete the scan and its artifacts |
| `GET` | `/api/scans/{id}/floorplan` | The floorplan schema |
| `GET` | `/api/scans/{id}/colliders` | Physics colliders |
| `GET` | `/api/scans/{id}/artifacts/{name}` | SVG · DXF · PDF · JSON · OBJ · GLB · PLY |
| `GET` | `/api/scans/{id}/frames/{file}` | A captured RGB frame |

`/api/scan/*` (singular) still works and resolves to the most recent finished scan, so existing
scripts do not break.


---

## Viewer

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

