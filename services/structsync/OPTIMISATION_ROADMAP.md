# Realism optimisation roadmap

Hand-off document for the image-quality work on the StructuralSync prototype. Read this first, then
`README.md` → "Realism notes". Last updated 2026-11-14.

## 1. Where things stand

### Shipped and verified (5 sample photos, before/after crops checked at full resolution)
| Change | Where | Evidence |
|---|---|---|
| Shading = large-scale light × fine *darkening* detail (grooves, bevels, contact shadows); detail auto-attenuated on patterned originals | `render.py` `shading_layer` | cabinets keep door panels/handles; no mosaic/granite leak (kitchen3 crops) |
| Highlight clamp + log-domain gain limit on dark originals | `render.py` `shading_layer` | no reflection streaks on dark island top |
| Soft highlight shoulder (knee 0.7 linear) | `render.py` `highlight_shoulder` | kitchen3 Calacatta: 24 % clipped → 0 % |
| Guided-filter edge snapping, **hard** composite | `render.py` `guided_alpha`, `project_materials` | renderer-only edge recall 0.995 hard vs 0.980 soft; no halo |
| Trilinear mips, √cos footprint | `render.py` `sample_texture`, `surfaces.py` `compute_uv` | no mip banding on floors |
| Counter/floor grain aligned to backsplash/cabinet/wall | `surfaces.py` `align_horizontal_axes` | visual |
| Seamless procedural catalogue (marble, granite, wood, tile, paint) | `catalog.py` | 2×2 tiling check |
| Handles/knobs as occluders (with 1 % area guard) | `vision.py` | visual |
| "upper/base cabinet" in a **separate** DINO pass, kept only if overlapping a door/drawer box | `vision.py` `detect` | kitchen3 uppers fully covered; no beam/shelf false positives |
| Drop cabinet boxes whose SAM mask fills < 50 % of the box | `vision.py` `parse` | removes stool/table-leg paint |
| Drop auto cabinet components with solidity < 0.75 | `surfaces.py` `drop_ragged` (called in `pipeline._orient`) | removes kitchen4/5 + kitchen3 high-chair speckle |
| SAM decodes boxes in chunks of 8 | `vision.py` `masks_from_boxes` | fixed 8 GB OOM |
| Full-resolution render (analysis ≤1024, render ≤2048) | `pipeline.py` `render_surfaces` | 1:1 render bit-identical; areas + detail_keep unchanged; visibly sharper |
| Countertop front edge (geometry-derived slab face) | `surfaces.py` `front_edge`, `pipeline._edges` | kitchen1/2/3 edges correct; SKU ranks unchanged; areas unchanged |

Metrics (eval.py, same machine): baseline SKU 53/66, edge recall 0.983 → isolated renderer changes SKU 59/66,
edge recall 0.978; outside_change = 0 throughout. **Thresholds (fill 0.5, solidity 0.75, detail knee
0.015–0.065) were tuned on 5 photos only.** kitchen4 and kitchen5 now auto-detect no cabinet (only fragments
existed) → SKU denominator 66 → 60.

### Written but NOT fully verified
| Change | Status | What's missing |
|---|---|---|
| Planar reflections (`render.py` `planar_reflections`, default **off**) | runs (0.4 s, kitchen1) | never visually inspected on all 10 renders |
| Camera grain matching (`render.py` `match_grain`, default **on**) | verified: eval unchanged (SKU 58/60, edge recall 0.973); 3× zoom kitchen1/kitchen3 looks like the photo's noise (σ≈0.0015–0.004) | grain flag not in `eval.py` |
| Harmonizer fallback (load/run failure → projection + note) | verified: falls back instantly, no download (`local_files_only`) | harmonizer itself untested (8 GB GPU, weights incomplete, low RAM) |
| Edge pixel constants scaled by render width (`px_scale`) | verified: 2× input → edge px ×2.4 (expected 2.56), BOM 2.34 vs 2.32 m² | — |
| Switches: `Engine(edges, reflections, grain)`, `render(..., edges=, reflections=, grain=)`, Gradio + web UI toggles, `eval.py --no-edges --reflections`, `smoke_test.py` flags | verified headless + via web API/UI | grain flag not yet in `eval.py` |

### Web app (2026-11-14)
`web/server.py` (FastAPI) + `web/static` (no-build SPA) + `Dockerfile` / `docker-compose.yml` / `run_web.sh`.
Verified: 40/40 API end-to-end checks (auth, uploads, analysis job, tap add/remove, renders with every switch,
catalogue CRUD + embedding rebuild, restart persistence, deletes, path traversal) and a scripted Firefox session
(projects → surfaces → tap → render → BOM/history → catalogue → add dialog; no JS errors).
**Docker image not built yet** — run `docker compose build` on a host with the NVIDIA container toolkit.

## 2. Next optimisations (ranked by realism per effort)

### A. Normal + roughness maps — WRITTEN, NOT RUN ON PHOTOS (2026-11-14)
- `catalog.py`: `SKU.normal` / `SKU.roughness` (None when absent); tile + wood generators fill height/roughness
  without extra random draws (all 12 albedos verified byte-identical); `height_to_normal`, `write_demo_maps`
  (only creates missing maps, skips any SKU whose albedo isn't the demo one). **Not yet run on the live catalogue**:
  `python -m structsync.catalog --write-demo-maps` (adds maps for CT-OAK-04, CB-WAL-04, BS-SUB-01, BS-GRN-02).
- `render.py`: `sample_map` (generic trilinear sampler; albedo path unchanged), `relief_shading` (per-pixel plane
  basis; light = up + 0.5·n_out — *changed from the original design* so a flat map is an exact identity on side
  walls too), roughness → specular weight. Skipped on derived `_edge` surfaces.
- Switch: `Engine(material_maps=False)`, `render(material_maps=)`, web `RenderReq.material_maps` + UI toggle
  "Surface relief", `eval.py --material-maps`, `smoke_test.py --material-maps`. Web: optional normal/roughness
  upload on create, `POST /api/catalog/{id}/maps`, `DELETE /api/catalog/{id}/maps/{kind}`, `GET .../map/{kind}`.
- Verified numerically only (`../_checkpoint_v6/test_maps.py`, synthetic planes): defaults bit-identical to v5;
  flat map = identity on camera-facing and side walls; bump brightness tracks up-facing slope (corr 0.97);
  roughness 0.5 keeps gloss. Note: a steep test sinusoid gave micro-shading std 0.29 — `max_tilt=0.6` may be too
  strong on real tiles; tune on photos.
- **First-run checks**: write demo maps → eval with and without `--material-maps` (areas/SKU/edge recall must not
  regress, outside_change 0) → 3× zoom on subway backsplash + oak floor; look for grout lines that read as relief,
  not as dark stripes; check side walls aren't darker overall.

### B. Real material scans
Replace procedural SKUs with supplier scans (or CC0 sets — ambientCG, Poly Haven — for demos):
albedo + normal + roughness at ≥2K, `tile_w/tile_h` = real metres. Delete `catalog/siglip_embeddings.npy` after.

### C. Mask quality
- Full-res render shows a small stair-step along low-contrast diagonal mask edges (kitchen1 cabinet top vs soffit):
  the 1024-px mask is upsampled and the guided snap has no image edge to follow there. Fix: upsample the SAM
  logits (not the binary mask) or run SAM at render resolution for the final masks.
- SAM 2.1 large: env switch `STRUCTSYNC_SAM=facebook/sam2.1-hiera-large` added (default unchanged; weights must be
  downloaded first). Compare per-surface areas + overlays on all samples before changing the default.
- Stair-step fix (logit upsampling) deliberately deferred: it sits in the model path and can't be validated without running SAM.
- Matting (e.g. ViTMatte) for hair-thin occluders (pendant cords, faucets) — rims around the kitchen3 pendants
  come from occluder masks larger than the object.
- Second indoor semantic segmenter as a vote (check licence: project requires commercially permissive models).

### D. Learned intrinsic decomposition
Replace heuristic shading split with a learned albedo/shading/specular model (licence check first). Removes old
reflections and patterns without hand thresholds.

### E. Per-surface colour matching — WRITTEN, NOT RUN ON PHOTOS (2026-11-14)
`render.py` `neighbour_tint`: illuminant colour from a ring of untouched pixels around each surface (all re-textured
surfaces excluded), saturation < 0.4 and mid luminance only, shift capped at ±6 % of the global gray-world tint.
Switch `local_tint` (Engine / render / web toggle "Local light colour" / `eval.py --local-tint`), default off.
Synthetic checks: red neighbour → no change; warm-lit neutral → capped shift. Contrast matching not done.
**First-run checks**: kitchen2 (red wall) and kitchen3 (teal island) must not tint the new stone; compare SKU rate.

### F. Generative refinement (needs ≥24 GB GPU)
Whole-surface low-strength pass with depth + line ControlNets and the SKU texture as image prompt (IP-Adapter);
keep drift-ΔE and SKU-recognition gates. Seam-only SDXL exists but is untested.

### G. Evaluation
- Done: `eval.py --no-grain --material-maps --local-tint`; switch states are written as columns in results.csv.
- 30–50 varied photos (every fix this session broke a different photo; 5 is not enough).
- Add a realism signal: blind A/B, or edit-detectability score.

### Deferred (not code-only)
- B real scans (assets), D intrinsic decomposition (needs a commercially-licensed model choice), F generative
  refinement (needs ≥24 GB GPU).
- Note: saved web projects can be re-rendered with new features through `Engine.restore` (scene arrays on disk) —
  no need to re-run detection to try A/E on existing projects.

## 3. Test checklist (run after any change)
1. `smoke_test.py`; headless `app.do_analyze`, `do_click` (add + remove), `do_render` with each switch combination.
2. `eval.py samples/ --renders 3` with defaults, then `--no-edges`, `--reflections`. Gate: per-surface areas
   unchanged vs previous run, SKU pass rate not lower, `outside_change` = 0.
3. Renderer-only comparison: old catalogue + original prompts (see scratch `iso.py` pattern: patch
   `structsync.catalog.ROOT` and `vision.PROMPTS` before importing eval).
4. One sample upscaled 2× → edge px ratio ≈ (2048/1280)², BOM areas identical.
5. Open **all 10** before/after renders at full resolution (`outputs/realism_compare/`); wrong paint is worse than a miss.
6. Time on a quiet machine (load average was ~12 during the last timings; full-res swap ≈ 1.6× projection cost).

## 4. Pitfalls learned
- Grounding DINO is prompt-context sensitive: adding phrases to a pass changes detections of the *other* phrases.
  Always check per-surface areas after any prompt change.
- `verify` uses fixed pixel constants → always measure at analysis resolution.
- Harmonizer must never auto-download (now `local_files_only` unless `STRUCTSYNC_ALLOW_DOWNLOAD=1`).
  A partial ControlNet download (~2.7 GB) may be left in `~/.cache/huggingface/hub/models--diffusers--controlnet-depth-sdxl-1.0`.
- No git in this folder: checkpoints are in `../_backup` (original), `../_baseline_tree` (runnable original),
  `../_checkpoint_v1..v3`.
