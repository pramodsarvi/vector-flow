# Plan: phone photos → Gaussian splat in the browser

*Status: proposal · 2026-11-20*

## Goal

A user opens a page on their phone, takes or picks a series of photos (or a
short video) of a room, and uploads them. A few minutes later anyone with access
can open the result in a desktop or phone browser and orbit or walk through a
photoreal 3D Gaussian Splatting (3DGS) reconstruction.

## What exists today

| Piece | State |
|---|---|
| Scanner `engine/export/splatting.py` | Writes `3dgs/transforms.json` from a **LiDAR** scan bundle that already has camera poses. Does not train. |
| Scanner upload | Accepts `.ply` (required) + `camera_poses.json` + frames. Plain photos cannot be processed. |
| iOS capture (`capture/ios/RoomScannerSession.swift`) | Records keyframes and poses, does not save depth; cannot yet produce a bundle. |
| StructuralSync | Single photo → material swap. No multi-view reconstruction. |
| Hosting | One RunPod pod (RTX PRO 4500, 32 GB VRAM), public via the 8080 gateway (`deploy/runpod/`). |

Missing: estimating camera poses from ordinary photos, training, splat storage
and a splat viewer.

## Proposed design

A new GPU service, `services/splat/`, built the same way as the other two
(FastAPI, plain HTML/JS frontend, no build step, `.env` token auth). It runs on
`127.0.0.1:8100` and the gateway exposes it at `/splat/`.

```text
phone page ──upload──▶ ingest ──▶ poses ──▶ train ──▶ export ──▶ viewer
 photos / video        frames     SfM       gsplat    .ply/.spz   Spark (three.js)
                                                         │
                       job queue + progress polling ◀────┘
```

### 1. Capture and upload (phone)

- Mobile-first page at `/splat/` with
  `<input type="file" accept="image/*,video/*" multiple capture>`.
- Resize photos in the browser (long edge ≈ 1600 px, JPEG q≈0.9) before upload:
  this cuts a 150-photo upload from ~600 MB to ~60 MB and gets past mobile data
  limits.
- Upload in chunks with retry, so a dropped connection does not restart the
  whole series.
- On-screen capture tips: walk around slowly, keep 60–70 % overlap between
  shots, avoid motion blur, include textured surfaces, 50–200 photos per room.

### 2. Ingest

- Video: extract frames with `ffmpeg` at 2–3 fps.
- Drop blurry frames (variance of Laplacian) and near-duplicates.
- Fix EXIF orientation; read focal length from EXIF as an intrinsics prior.

### 3. Camera poses

| Option | Licence | Notes |
|---|---|---|
| **COLMAP + GLOMAP** (default) | BSD | Proven; GLOMAP global mapping is 10–50× faster than incremental COLMAP. A few minutes for 150 images on GPU. |
| VGGT (feed-forward) | Check: research weights are non-commercial; verify the terms of any commercial checkpoint before use | Seconds instead of minutes, and more robust on low-texture rooms. Optional second path. |
| MASt3R / DUSt3R | CC BY-NC | Non-commercial; **not usable** given the repo's commercial-permissive rule. |

Scans from the Scanner bundle that already carry poses skip this stage and reuse
`transforms.json`.

### 4. Training

- **gsplat** (Apache-2.0, nerfstudio project) as the trainer. The original
  Inria 3DGS code is non-commercial and is not used.
- Defaults: 7k iterations for a fast preview (~3–5 min), 30k for the final
  result (~15–25 min) on the RTX PRO 4500. Initialise from the SfM sparse points
  (or the LiDAR cloud when present).
- Use the MCMC densification strategy with a Gaussian cap (~1–1.5 M) so results
  stay small enough for phone browsers.

### 5. Export

- `scene.ply` (standard 3DGS format) for download and other tools.
- `scene.spz` (compressed, ~10× smaller) for web viewing.
- A thumbnail and the camera path for the viewer's initial view.

### 6. Viewer

- **Spark** (MIT, three.js-based) renders `.ply`/`.spz`/`.splat` in WebGL2 and
  works on mobile Safari and Chrome. GaussianSplats3D (MIT) is the fallback.
- Vendored into the service like the Scanner's three.js (no CDN at runtime).
- Orbit + first-person controls, share link, download button.

### Jobs and API

The RunPod proxy cuts requests at ~100 s, so every heavy stage runs as a
background job and the browser polls.

```text
POST   /api/captures                 create a capture, returns id
PUT    /api/captures/{id}/files      chunked upload
POST   /api/captures/{id}/process    queue ingest → poses → train → export
GET    /api/captures/{id}            status, stage, progress %, log tail
GET    /api/captures/{id}/scene.spz  result (token via header or ?token=)
DELETE /api/captures/{id}
```

- One training job at a time (a single GPU); the rest wait in a FIFO queue.
- GPU sharing with StructuralSync: it keeps its models resident (~5 GB+), and
  gsplat for a room needs ~8–16 GB, so both fit in 32 GB. Cap the Gaussian count
  and fail cleanly on out-of-memory errors.
- Storage under `/workspace/splat-data/` (needs a Volume Disk to persist).

## Milestones

| # | Deliverable | Estimate |
|---|---|---|
| 1 | Service skeleton, token auth, phone upload page, job queue, gateway route | 0.5 day |
| 2 | Ingest + COLMAP/GLOMAP poses, with sparse point cloud preview | 0.5–1 day |
| 3 | gsplat training + `.ply`/`.spz` export, progress reporting | 1 day |
| 4 | Spark viewer page, share link, mobile testing | 0.5 day |
| 5 | Reuse Scanner LiDAR bundles (skip poses, metric scale) | 0.5 day |
| 6 | Tests (fixture image set), README, RunPod notes | 0.5 day |

Milestones 1–4 give a working end-to-end flow (~3 days); 5–6 harden it.

## Risks

- **Plain white walls** give SfM too few features, so reconstruction breaks up.
  Mitigation: capture guidance, try VGGT if licensing allows, LiDAR path from
  milestone 5.
- **No metric scale** from photos alone. Fine for viewing; measurement needs
  LiDAR or a known reference length.
- **Phone browser memory**: large splats crash mobile Safari. Mitigation:
  Gaussian cap and `.spz`.
- **Pod is not persistent**: without a Volume Disk every restart loses captures
  and results.
- **Public URL**: the service must require its token before the gateway route
  is added.

## Open questions

1. Is commercial-permissive licensing a hard requirement for this service too?
   (It rules out MASt3R and the Inria trainer, and decides VGGT.)
2. Photos only, video only, or both?
3. Should results be shareable without a token (signed read-only links)?
4. Should this live inside the Scanner viewer as a third view, or stay a
   separate page?
