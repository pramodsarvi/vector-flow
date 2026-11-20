# StructuralSync prototype

Single photo + fixed catalogue → kitchen finish swap. Implements the v2 plan
(`../ai_renovation_engine_plan_v2.md`) end to end, using only commercially permissive models.

## Quick start (Linux + NVIDIA GPU)
The backend (FastAPI) serves the frontend (plain HTML/JS/CSS in `web/static/`) — **one process, no Node/npm, no separate build.**

```bash
# 1. one-time setup: venv + pinned dependencies + model weights (~13 GB, needs internet)
bash setup_gpu.sh --download-models

# 2. run (backend API + web UI together)
./run_web.sh                  # →  open http://127.0.0.1:8000   (API docs: /docs)
```
(`make setup` / `make run` / `make test` are shortcuts for the same commands if `make` is installed.)
Stop with `Ctrl-C`. Options: `./run_web.sh --preload` (load models at start), `--public` (listen on all interfaces —
set `STRUCTSYNC_TOKEN` first), `PORT=8001 ./run_web.sh` (other port). Copy `.env.example` to `.env` for settings.
First analysis after a cold start loads the models (extra ~30 s); later ones take 20–45 s, renders 1–5 s.

**RunPod:** see [RUNPOD.md](RUNPOD.md).  
**Docker (whole stack, one command):** `docker compose up -d --build` → http://127.0.0.1:8000 (needs NVIDIA driver ≥ 580 + container toolkit; first start downloads ~10 GB of weights into a volume). `docker compose down` stops it.

**Check it works:** `uv pip install -p .venv/bin/python -r requirements-dev.txt && .venv/bin/python -m pytest` (118 tests, no GPU needed), then in the UI create a project and upload
a photo from `samples/`.

Troubleshooting: *".venv not found"* → run `bash setup_gpu.sh --download-models`. *"port in use"* → `PORT=8001 ./run_web.sh`.
*CUDA out of memory* → set `STRUCTSYNC_SCENES=1`, close other GPU apps.

## Web app (projects · catalogue · render history)
```bash
./run_web.sh                      # http://127.0.0.1:8000   (uses .venv; --public binds 0.0.0.0)
# or with Docker (NVIDIA container toolkit required):
docker compose up -d --build      # http://127.0.0.1:8000
```
- **Projects**: upload a photo → surfaces detected in the background → click the surface overlay to add/remove regions →
  pick materials (or describe one, e.g. "dark marble") → render → before/after slider, bill of materials, checks.
  Every render is kept in the project's history (download, revisit, delete).
- **Surfaces**: six classes — countertop, cabinet, backsplash, floor, wall and **ceiling**. The
  orientation filter is signed, not just "is it horizontal": a floor and a ceiling are the two
  largest horizontal planes in a room, and telling them apart is the whole difference between
  painting the ceiling and painting the floor. Non-kitchen rooms are handled through the
  **occluder** vocabulary (beds, sofas, wardrobes, curtains, mirrors, rugs, bathroom fixtures) —
  what makes a bedroom photo usable is not the surface list but whether the furniture standing in
  front of the wall is protected from the wall's new paint.
- **Regions**: a detected surface is addressable per instance, not per class, so two walls or two
  cabinet doors can take different materials. A region id is `<kind>#<n>` (`wall#1`); the materials
  panel lists one row per region, and the correction-target picker chooses which one a click edits.
  `assign` still accepts a bare class name (`{"wall": "WL-GRG-01"}`), which covers every region of
  that class. How a class divides depends on the class: walls split by plane (so a fridge standing
  in front of one doesn't make it two), counters and backsplashes by connected run, cabinets per
  detected door, and the floor never splits. Ids are assigned once per analysis, largest region
  first, and stay fixed through corrections and restarts, so a stored assignment and its cache key
  keep resolving. Projects analysed before regions existed load as one region per class.
- **Catalogue**: add / edit / delete materials, upload or replace textures; search embeddings rebuild automatically.
- Data lives in `STRUCTSYNC_DATA` (default `./data`, one folder per project — no database, no user accounts).
  `STRUCTSYNC_CATALOG` moves the catalogue; `STRUCTSYNC_TOKEN` requires a shared token on every API call
  (set it before exposing the server beyond localhost). API docs: `/docs`.
- One process drives the single GPU; requests queue for it. Analysis ≈ 20–45 s, renders ≈ 1–5 s.
- **Result cache** (`STRUCTSYNC_DATA/cache`, cap `STRUCTSYNC_CACHE_MB`, default 4096, `0` = off): the same photo is
  analysed only once (any project), and the same photo + corrections + materials + options is rendered only once —
  repeats return instantly (marked "cached"). Keys include the pipeline source/model version and each SKU's
  catalogue entry and texture pixels, so code, price or texture changes recompute automatically. "Re-analyse"
  always recomputes. The cache is disposable (`DELETE /api/cache` or delete the folder). A cached render's
  SKU-recognition report reflects the catalogue when it was first rendered.

## Run on a GPU box (Linux + NVIDIA)
```bash
bash setup_gpu.sh --download-models --with-harmonizer   # venv + deps + ~13 GB weights
.venv/bin/python smoke_test.py --harmonize              # one photo end to end → outputs/
.venv/bin/python eval.py samples/ --harmonize           # batch → outputs/eval/results.csv + grids
.venv/bin/python app.py --host 0.0.0.0 --preload        # UI on :7860  (or --share for a public URL)
```
Drop more kitchen photos into `samples/` (or any folder) for `eval.py`. Its summary line prints the
median analyse time, swap p50/p90, SKU recognition pass rate and max outside-mask change.

## Run on the Mac (MPS)
```bash
.venv/bin/python app.py          # UI at http://127.0.0.1:7860
.venv/bin/python smoke_test.py   # headless run on samples/kitchen1_ccby.jpg → outputs/
```
Setup used: `uv venv -p 3.12 .venv`, then torch, transformers, diffusers, gradio, opencv-python-headless, scipy,
and MoGe pinned to `git+https://github.com/microsoft/MoGe.git@0744441` (the V3 release needs Triton, which has no macOS build).

## Development
```bash
uv pip install -p .venv/bin/python -r requirements-dev.txt   # pytest, ruff (pinned)
.venv/bin/python -m pytest            # 118 tests, ~90 s, never loads a model (model constructors are patched to raise)
.venv/bin/python -m ruff check .      # lint (config in pyproject.toml)
```
- Dependencies: `requirements.txt` = pinned top-level packages; `requirements.lock` = exact verified environment
  (used by `setup_gpu.sh` and the Dockerfile). Upgrade deliberately: bump, run tests + `eval.py samples/`, re-lock.
- Logging: library modules use `logging.getLogger(__name__)`; the web server's level is `STRUCTSYNC_LOG` (default INFO).
- Tests cover renderer behaviour on synthetic planes, catalogue determinism, geometry, and the whole web API on a
  synthetic room. They can't judge realism on photos — use `eval.py` and `outputs/realism_compare/` for that.

## Pipeline → code
| Plan stage | File | Models |
|---|---|---|
| 0 Catalogue prep + search | `structsync/catalog.py` | SigLIP 2 (Apache-2.0) |
| 1 Scene parse + tap-to-correct | `structsync/vision.py` `Segmenter` | Grounding DINO, SAM 2.1 (Apache-2.0) |
| 2 Geometry solve | `structsync/vision.py` `Geometry` | MoGe-2 ViT-L (MIT) |
| 3 Surface model (signed orientation filter, RANSAC planes, metric UVs, area, height priors) | `structsync/surfaces.py` | none (pure geometry) |
| 5 Texture projection + shading transfer | `structsync/render.py` | none (deterministic) |
| 6 Seam harmonisation (optional) | `structsync/render.py` `Harmonizer` | SDXL-Inpainting + depth ControlNet (OpenRAIL++), depth from MoGe |
| 7 Verification + fallback | `structsync/render.py` `verify`, `drift_deltaE`, `sku_recognition` | SigLIP 2 |

## Catalogue
`catalog/catalog.json` + `catalog/<SKU>/albedo.png` (optional `normal.png`, `roughness.png`). The 12 demo SKUs are
procedurally generated placeholders; `tile_w`/`tile_h` = metres one texture tile covers.

**Bulk import / export (web UI → Catalogue → Import ZIP / Export, or `POST /api/catalog/import`)**
- One ZIP: `catalog.json` *or* `catalog.csv` (columns `id,name,category,vendor,price,tile_w,tile_h,gloss`; comma,
  semicolon or tab; optional `albedo`/`normal`/`roughness` columns naming files in the ZIP) plus
  `<id>/albedo.png|jpg` per SKU. A wrapper folder, `__MACOSX/` and dot-files are fine. **Export produces exactly
  this format**, so the easiest template is an export.
- **Add & update** (default): new SKUs are added, listed ones updated, others kept; an existing SKU without an image
  in the ZIP keeps its texture (a CSV alone updates prices). **Replace**: the catalogue becomes exactly the ZIP.
- All-or-nothing: every row and image is checked first and all problems are listed with row numbers; the UI shows
  the result (added / updated / removed) before you confirm. Applying is crash-safe; an interrupted import is
  finished or rolled back at the next start. Limit `STRUCTSYNC_MAX_IMPORT_MB` (default 2048).
- Replacing a texture drops that SKU's old normal/roughness maps. Search embeddings rebuild automatically.

## Measured on Apple M5 (MPS), 1024 px photo
- Analyse (once per photo): ~25 s (segmentation ~12 s, MoGe-2 ~10 s, surfaces ~2 s) — measured
  before the vocabulary grew to cover rooms other than kitchens, which took Grounding DINO from 4
  passes to 7; expect segmentation to roughly double and re-measure on a real GPU.
- Each material swap (projection path): ~0.3–0.5 s

## Verification (shown in the UI and in results.csv)
- **outside_change**: mean pixel change outside the edited surfaces (should be ≈0).
- **boundary_edge_recall**: share of structural edges at surface boundaries that survive the edit.
- **drift ΔE** (harmonise only): median ΔE between the harmonised output and the deterministic projection.
  Above 6, the harmonised result is rejected and the projection is kept.
- **SKU recognised**: each edited surface is cropped from the final render, embedded with SigLIP 2, and ranked
  against all same-category SKUs. Pass = the assigned SKU ranks #1. This is independent of the renderer.

## Realism notes (Nov 2026 audit)
- **Render resolution**: analysis (models, masks, tap-to-correct) runs at ≤1024 px; textures are re-projected
  analytically at ≤2048 px (`RENDER_MAX_SIDE`) with masks snapped onto the full-res photo. Metrics are measured at
  analysis resolution; BOM areas come from the analysis-res surfaces. The SDXL harmoniser still works at ≤1024 px and
  only its seam band is blended back into the full-res render.
- **Shading** = large-scale light (masked low-pass) × fine *darkening* detail (door grooves, bevels, handle/contact
  shadows). Detail is attenuated automatically when the original surface was patterned, so old patterns don't leak.
  New-material highlights roll off softly above 0.7 linear so white stone keeps its veins in bright photos.
- **Edges**: guided-filter snapping onto real image edges, hard composite (no halo). Handles/knobs are occluders.
- **Countertop front edges** (`edges=True`, UI checkbox, `--no-edges`): the slab's front face is derived from geometry
  (3 cm drop along gravity, point-map agreement, stops at occluders) and gets the top's SKU with the texture wrapped
  over the edge and the original face/top brightness ratio. Never counted in BOM areas.
- **Mirror reflections** (`reflections=False` by default, UI checkbox, `--reflections`): screen-space ray march through
  the point map on glossy (gloss ≥ 0.5) horizontal SKUs, Schlick Fresnel weighted, roughness-blurred. Misses add
  nothing. Experimental: a single view can't see what a mirror would show.
- **Grain** on counters/floors follows the backsplash/cabinet/wall; textures use trilinear mips.
- **Cabinet detection**: "upper/base cabinet" phrases run in their own Grounding DINO pass (adding them to the surface
  prompt made DINO drop whole walls) and are kept only when they overlap a door/drawer box; cabinet boxes whose SAM
  mask fills <50 % of the box, and auto-detected cabinet components with solidity <0.75, are dropped (they rendered
  as speckled paint). These thresholds were tuned on the 5 sample photos only. Consequence: kitchen4 and kitchen5
  now auto-detect **no** cabinet (their only candidates were fragments) — use tap-to-correct to add them.
- **Harmoniser**: on GPUs under 14 GB it uses model CPU offload; if it can't load or run (missing weights, OOM) the
  render falls back to the projection with a note. Untested end to end on the 8 GB dev box (weights not downloaded,
  not enough free RAM).
- **Catalogue**: the demo textures are seamless procedural marble/granite/wood/tile/paint. Real supplier scans are
  still the biggest remaining realism win.
- Before/after comparisons: `outputs/realism_compare/`.

## Known limitations
- Segmentation misses some regions (e.g. peninsula front panel, parts of the back counter). Fix them with tap-to-correct.
- No floor in view → no height prior at all; MoGe metric depth is used as-is (areas ±20 %). With a
  floor but no counter (a bedroom, hallway, bathroom) the ceiling-to-floor pair is used instead — a
  looser prior at 2.5 m, weighted below the counter's 0.91 m when both are visible, and rejected if
  the measured gap falls outside 1.8–4.5 m.
- **Rooms other than kitchens are untested on real photographs.** Ceiling, the wider occluder
  vocabulary and the ceiling height prior are covered by the synthetic test suite only; the
  detector thresholds are still the ones tuned on the 5 kitchen samples. Run `eval.py` against a
  bedroom/bathroom set before trusting the output there.
- Very strong old reflections on glossy originals are clamped but not fully removed.
- Cabinet door-style changes, furniture removal, and adding geometry are out of scope (per plan).
