"""StructuralSync web backend: projects, catalogue management and render history over a JSON API,
plus the static single-page UI in web/static.

Run:   .venv/bin/python -m uvicorn web.server:app --host 127.0.0.1 --port 8000
Env:   STRUCTSYNC_DATA     projects directory            (default ./data)
       STRUCTSYNC_CATALOG  catalogue directory           (default ./catalog)
       STRUCTSYNC_TOKEN    shared access token; if set every /api call needs it
                           (header "Authorization: Bearer <token>" or ?token=<token> for <img> URLs)
       STRUCTSYNC_SCENES   scenes kept in memory         (default 2; each ≈ 100–200 MB)
       STRUCTSYNC_PRELOAD  "1" to load models at start-up

One process, one GPU: every model/engine call goes through a single-worker executor, so requests
never run the (non-thread-safe) Engine concurrently. Analysis runs as a background job the UI polls.
No accounts or database by design: projects are folders under STRUCTSYNC_DATA.
"""
from __future__ import annotations

import io
import json
import logging
import os
import re
import shutil
import sys
import threading
import time
import uuid
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel, Field
from starlette.background import BackgroundTask

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from structsync import catalog as C  # noqa: E402
from structsync import catalog_io as CIO  # noqa: E402
from structsync.pipeline import Engine  # noqa: E402
from structsync.regions import kind_of  # noqa: E402
from structsync.vision import SURFACES  # noqa: E402
from structsync.viz import COLORS, overlay, region_color  # noqa: E402
from web.cache import Cache, file_sha, scene_rev  # noqa: E402

DATA = Path(os.environ.get("STRUCTSYNC_DATA", ROOT / "data")).resolve()
(DATA / "projects").mkdir(parents=True, exist_ok=True)
TMP = DATA / ".tmp"                  # uploads/exports in flight; cleared at start-up
TOKEN = os.environ.get("STRUCTSYNC_TOKEN") or None
MAX_IMPORT = int(os.environ.get("STRUCTSYNC_MAX_IMPORT_MB", "2048")) * 2 ** 20
MAX_SCENES = int(os.environ.get("STRUCTSYNC_SCENES", "2"))
MAX_UPLOAD = 25 * 2 ** 20           # bytes
MAX_PIXELS = 40_000_000
Image.MAX_IMAGE_PIXELS = MAX_PIXELS
ID_RE = CIO.ID_RE
PID_RE = re.compile(r"^[a-f0-9]{12}$")

logging.basicConfig(level=os.environ.get("STRUCTSYNC_LOG", "INFO"),
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("structsync.web")

engine = Engine()
cache = Cache(DATA / "cache")
gpu = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gpu")   # the only place the Engine runs
_scenes: OrderedDict[str, object] = OrderedDict()
# Re-entrant: update_meta() holds it across a read-modify-write and calls save_meta() inside.
_meta_lock = threading.RLock()
# _scenes is touched from both the GPU worker and the request thread pool.
_scenes_lock = threading.Lock()


def on_gpu(fn, *a, **kw):
    """Run on the single engine worker and wait (FastAPI runs sync endpoints in a thread pool)."""
    return gpu.submit(fn, *a, **kw).result()


# ------------------------------------------------------------------ auth
def auth(request: Request):
    if TOKEN is None:
        return
    h = request.headers.get("authorization", "")
    got = h[7:] if h.lower().startswith("bearer ") else request.query_params.get("token")
    if got != TOKEN:
        raise HTTPException(401, "missing or wrong access token")


@asynccontextmanager
async def lifespan(_app):
    shutil.rmtree(TMP, ignore_errors=True)
    TMP.mkdir(parents=True, exist_ok=True)
    for msg in CIO.recover(C.ROOT):
        log.warning("catalogue import recovery: %s", msg)
    # projects left "analyzing" by a crash/restart would spin forever: re-queue them
    for d in (DATA / "projects").iterdir():
        try:
            m = json.loads((d / "project.json").read_text())
        except Exception:  # noqa: BLE001
            continue
        if m.get("status") == "analyzing":
            log.info("re-queueing interrupted analysis %s", m["id"])
            _submit_analysis(m["id"])
    if os.environ.get("STRUCTSYNC_PRELOAD") == "1":
        gpu.submit(engine.preload)
    log.info("data=%s catalog=%s auth=%s device=%s", DATA, C.ROOT, TOKEN is not None, engine.device)
    yield
    gpu.shutdown(wait=False, cancel_futures=True)


app = FastAPI(title="StructuralSync", version="1.0", lifespan=lifespan)
api = Depends(auth)


@app.exception_handler(Exception)
async def _unhandled(request, exc):  # keep the UI informed instead of a bare 500
    log.exception("unhandled error on %s %s", request.method, request.url.path, exc_info=exc)
    return JSONResponse({"detail": f"{type(exc).__name__}: {exc}"}, status_code=500)


# ------------------------------------------------------------------ helpers
def read_image(raw: bytes) -> Image.Image:
    if len(raw) > MAX_UPLOAD:
        raise HTTPException(413, f"file larger than {MAX_UPLOAD // 2 ** 20} MB")
    try:
        im = Image.open(io.BytesIO(raw))
        im.load()
    except Exception:  # noqa: BLE001  (includes decompression-bomb errors)
        raise HTTPException(400, "not a readable image (or too many pixels)") from None
    return im.convert("RGB")   # re-encode: strips metadata / non-image payloads


def png(arr_or_img, max_side=None) -> Response:
    im = arr_or_img if isinstance(arr_or_img, Image.Image) else Image.fromarray(arr_or_img)
    if max_side and max(im.size) > max_side:
        im = im.copy()
        im.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    im.save(buf, "PNG" if not max_side else "JPEG", quality=88)
    return Response(buf.getvalue(), media_type="image/png" if not max_side else "image/jpeg",
                    headers={"Cache-Control": "no-store"})


def pdir(pid: str) -> Path:
    if not PID_RE.match(pid):
        raise HTTPException(404, "no such project")
    d = DATA / "projects" / pid
    if not d.exists():
        raise HTTPException(404, "no such project")
    return d


def load_meta(pid: str) -> dict:
    return json.loads((pdir(pid) / "project.json").read_text())


def save_meta(pid: str, meta: dict):
    with _meta_lock:
        p = DATA / "projects" / pid / "project.json"
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(meta, indent=2))
        tmp.replace(p)


def update_meta(pid: str, mutate) -> dict:
    """
    Read-modify-write project.json while holding the lock.

    save_meta() alone only makes the *write* atomic, so a rename landing between
    another caller's load_meta() and its save_meta() was silently overwritten.
    Mutating through here closes that window.
    """
    with _meta_lock:
        meta = load_meta(pid)
        mutate(meta)
        save_meta(pid, meta)
        return meta


def save_state(pid: str, sc) -> str:
    """Persist the scene arrays; returns its revision (cache key part; changes with every correction)."""
    d = DATA / "projects" / pid
    st = engine.scene_state(sc)
    np.savez_compressed(d / "scene.tmp.npz", **st)
    (d / "scene.tmp.npz").replace(d / "scene.npz")
    return scene_rev(st)


def ensure_keys(pid: str, meta: dict) -> dict:
    """Older projects predate the cache: compute photo_sha / scene_rev from disk (no model needed)."""
    d = pdir(pid)
    if not meta.get("photo_sha"):
        meta["photo_sha"] = file_sha(d / "photo.jpg")
    if meta.get("status") == "ready" and not meta.get("scene_rev") and (d / "scene.npz").exists():
        with np.load(d / "scene.npz") as st:
            meta["scene_rev"] = scene_rev(dict(st))
    return meta


def remember(pid, sc):
    with _scenes_lock:
        _scenes[pid] = sc
        _scenes.move_to_end(pid)
        while len(_scenes) > MAX_SCENES:
            _scenes.popitem(last=False)


def forget_scene(pid):
    """Drop a cached scene. Safe to call from any thread."""
    with _scenes_lock:
        _scenes.pop(pid, None)


def scene_names() -> list[str]:
    """Snapshot for /api/health — iterating the live dict can race the GPU worker."""
    with _scenes_lock:
        return list(_scenes)


def get_scene(pid: str):
    """Scene from memory, or rebuilt from disk (runs on the GPU worker)."""
    with _scenes_lock:
        if pid in _scenes:
            _scenes.move_to_end(pid)
            return _scenes[pid]
    meta = load_meta(pid)
    if meta["status"] != "ready":
        raise HTTPException(409, f"project is {meta['status']}")
    d = pdir(pid)
    with np.load(d / "scene.npz") as st:
        sc = engine.restore(Image.open(d / "photo.jpg"), dict(st), meta.get("notes", []), meta.get("timings"))
    remember(pid, sc)
    return sc


def surfaces_summary(sc) -> list[dict]:
    """One entry per class, areas summed over its regions — the shape clients already expect."""
    out: dict[str, dict] = {}
    for rid, s in sc.surfaces.items():
        if not s.planes:
            continue
        k = kind_of(rid)
        e = out.setdefault(k, dict(kind=k, planes=0, area_m2=0.0, edge=False, regions=0))
        e["planes"] += len(s.planes)
        e["area_m2"] += s.area_m2
        e["edge"] = e["edge"] or rid in sc.edges
        e["regions"] += 1
    for e in out.values():
        e["area_m2"] = round(e["area_m2"], 2)
    return [out[k] for k in sorted(out)]


def regions_summary(sc) -> list[dict]:
    """One entry per addressable region: what the UI selects and assigns a material to."""
    return [dict(id=rid, kind=kind_of(rid), planes=len(s.planes), area_m2=round(s.area_m2, 2),
                 edge=rid in sc.edges, color=[round(c) for c in region_color(rid)])
            for rid, s in sc.surfaces.items() if s.planes]


def sku_json(s) -> dict:
    return dict(id=s.id, name=s.name, category=s.category, vendor=s.vendor, price=s.price,
                tile_w=s.tile_w, tile_h=s.tile_h, gloss=s.gloss,
                has_normal=(s.path / "normal.png").exists(), has_roughness=(s.path / "roughness.png").exists())


def reload_catalog():
    """After any catalogue write: the SigLIP cache only checks the SKU count, so delete it."""
    (C.ROOT / "siglip_embeddings.npy").unlink(missing_ok=True)
    engine.catalog = C.load_catalog()
    engine.by_id = {s.id: s for s in engine.catalog}
    engine._search = None


def write_catalog(entries: list[dict]):
    tmp = C.ROOT / "catalog.json.tmp"
    tmp.write_text(json.dumps(entries, indent=2))
    tmp.replace(C.ROOT / "catalog.json")


def harmonizer_weights_present() -> bool:
    try:
        from huggingface_hub import try_to_load_from_cache as cached

        from structsync.render import Harmonizer as H
        return all(isinstance(cached(r, f), str) for r, f in
                   [(H.MODEL, "unet/diffusion_pytorch_model.fp16.safetensors"),
                    (H.MODEL, "text_encoder_2/model.fp16.safetensors"), (H.VAE, "config.json"),
                    (H.CONTROLNET, "diffusion_pytorch_model.fp16.safetensors")])
    except Exception:  # noqa: BLE001
        return False


# ------------------------------------------------------------------ meta
@app.get("/api/health", dependencies=[api])
def health():
    import torch
    return dict(device=engine.device,
                gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                harmonizer=dict(weights_present=harmonizer_weights_present(),
                                loaded=engine._harm is not None, error=engine._harm_error),
                scenes_in_memory=scene_names(), auth=TOKEN is not None, cache=cache.stats())


@app.get("/api/config", dependencies=[api])
def config():
    return dict(surfaces=SURFACES, colors={k: "#{:02x}{:02x}{:02x}".format(*v) for k, v in COLORS.items()},
                switches=dict(edges=engine.edges, reflections=engine.reflections, grain=engine.grain,
                              material_maps=engine.material_maps, local_tint=engine.local_tint))


# ------------------------------------------------------------------ catalogue
@app.get("/api/catalog", dependencies=[api])
def catalog_list():
    return [sku_json(s) for s in engine.catalog]


@app.get("/api/catalog/search", dependencies=[api])
def catalog_search(q: str, category: str | None = None, k: int = 3):
    if category and category not in SURFACES:
        raise HTTPException(400, "unknown category")
    hits = on_gpu(engine.search.search, q, category, k)
    return [dict(sku_json(s), score=round(v, 3)) for s, v in hits]


@app.get("/api/catalog/{sid}/albedo", dependencies=[api])
def catalog_albedo(sid: str, size: int = Query(256, ge=32, le=2048)):
    s = engine.by_id.get(sid)
    if not s:
        raise HTTPException(404, "no such SKU")
    return png(Image.open(s.path / "albedo.png").convert("RGB"), size)


def _spool(upload: UploadFile, limit: int) -> Path:
    """Stream an upload to a temp file (never held in memory) with a size cap."""
    p = TMP / f"{uuid.uuid4().hex}.upload"
    n = 0
    with open(p, "wb") as f:
        while chunk := upload.file.read(2 ** 20):
            n += len(chunk)
            if n > limit:
                f.close()
                p.unlink(missing_ok=True)
                raise HTTPException(413, f"upload larger than {limit // 2 ** 20} MB")
            f.write(chunk)
    return p


@app.get("/api/catalog/export", dependencies=[api])
def catalog_export():
    """The whole catalogue as one ZIP (catalog.json + <id>/albedo|normal|roughness.png) — also the import
    template and a backup."""
    out = TMP / f"catalog-export-{uuid.uuid4().hex[:8]}.zip"
    on_gpu(CIO.export_zip, C.ROOT, out)
    return FileResponse(out, media_type="application/zip", filename=f"catalog-{time.strftime('%Y%m%d-%H%M')}.zip",
                        background=BackgroundTask(out.unlink, missing_ok=True))


@app.post("/api/catalog/import", dependencies=[api])
def catalog_import(archive: UploadFile = File(...), mode: str = Form("merge"), dry_run: bool = Form(True)):
    """Bulk add/update (merge) or replace the catalogue from one ZIP. All-or-nothing: every row and
    image is validated first. dry_run=true (default) only reports what would change."""
    if mode not in ("merge", "replace"):
        raise HTTPException(400, "mode must be merge or replace")
    path = _spool(archive, MAX_IMPORT)
    try:
        def work():
            plan = CIO.plan_import(path, C.ROOT, SURFACES, mode)
            if plan.errors or dry_run:
                return plan.summary() | dict(applied=False)
            CIO.apply_import(path, C.ROOT, plan)
            reload_catalog()
            log.info("catalogue import (%s): +%d ~%d -%d", mode, len(plan.added), len(plan.updated), len(plan.removed))
            return plan.summary() | dict(applied=True)
        res = on_gpu(work)
    finally:
        path.unlink(missing_ok=True)
    if not res["ok"] and not dry_run:
        return JSONResponse(res, status_code=422)
    return res


def _validate_sku(d: dict):
    errs = CIO.validate_entry(d, SURFACES)
    if errs:
        raise HTTPException(400, "; ".join(errs))


def _save_texture(sid: str, raw: bytes):
    im = read_image(raw)
    if min(im.size) < 64:
        raise HTTPException(400, "texture must be at least 64 px")
    im.thumbnail((2048, 2048), Image.LANCZOS)
    d = C.ROOT / sid
    d.mkdir(parents=True, exist_ok=True)
    im.save(d / "albedo.png")
    for k in ("normal", "roughness"):       # relief maps belonged to the old texture
        (d / f"{k}.png").unlink(missing_ok=True)


MAPS = {"normal": "RGB", "roughness": "L"}


def _save_map(sid: str, kind: str, raw: bytes):
    """Optional relief maps. Stored next to albedo.png; never part of catalog.json (SKU(**d) rejects
    extra fields) and never affect the SigLIP embeddings, so no reload is needed."""
    im = read_image(raw)
    if min(im.size) < 64:
        raise HTTPException(400, f"{kind} map must be at least 64 px")
    im.thumbnail((2048, 2048), Image.LANCZOS)
    im.convert(MAPS[kind]).save(C.ROOT / sid / f"{kind}.png")


@app.get("/api/catalog/{sid}/map/{kind}", dependencies=[api])
def catalog_map(sid: str, kind: str, size: int = Query(256, ge=32, le=2048)):
    s = engine.by_id.get(sid)
    if not s or kind not in MAPS or not (s.path / f"{kind}.png").exists():
        raise HTTPException(404, "no such map")
    return png(Image.open(s.path / f"{kind}.png").convert("RGB"), size)


@app.post("/api/catalog/{sid}/maps", dependencies=[api])
def catalog_set_maps(sid: str, normal: UploadFile | None = File(None), roughness: UploadFile | None = File(None)):
    files = {k: f.file.read(MAX_UPLOAD + 1) for k, f in (("normal", normal), ("roughness", roughness))
             if f and f.filename}
    if not files:
        raise HTTPException(400, "send a normal and/or roughness file")

    def work():
        if sid not in engine.by_id:
            raise HTTPException(404, "no such SKU")
        for k, raw in files.items():
            _save_map(sid, k, raw)
    on_gpu(work)
    return sku_json(engine.by_id[sid])


@app.delete("/api/catalog/{sid}/maps/{kind}", dependencies=[api])
def catalog_delete_map(sid: str, kind: str):
    def work():
        s = engine.by_id.get(sid)
        if not s or kind not in MAPS:
            raise HTTPException(404, "no such map")
        (s.path / f"{kind}.png").unlink(missing_ok=True)
    on_gpu(work)
    return sku_json(engine.by_id[sid])


@app.post("/api/catalog", dependencies=[api])
def catalog_create(id: str = Form(...), name: str = Form(...), category: str = Form(...),
                   vendor: str = Form(""), price: float = Form(...), tile_w: float = Form(...),
                   tile_h: float = Form(...), gloss: float = Form(0.3), albedo: UploadFile = File(...),
                   normal: UploadFile | None = File(None), roughness: UploadFile | None = File(None)):
    if not ID_RE.match(id):
        raise HTTPException(400, "SKU id: 2–32 letters, digits or '-'")
    d = dict(id=id, name=name, category=category, vendor=vendor, price=price, tile_w=tile_w,
             tile_h=tile_h, gloss=gloss)
    _validate_sku(d)
    raw = albedo.file.read(MAX_UPLOAD + 1)
    extra = {k: f.file.read(MAX_UPLOAD + 1) for k, f in (("normal", normal), ("roughness", roughness))
             if f and f.filename}

    def work():
        entries = json.loads((C.ROOT / "catalog.json").read_text())
        if any(e["id"] == id for e in entries):
            raise HTTPException(409, "SKU id already exists")
        _save_texture(id, raw)
        for k, b in extra.items():
            _save_map(id, k, b)
        write_catalog(entries + [d])
        reload_catalog()
    on_gpu(work)
    return d


class SKUUpdate(BaseModel):
    name: str | None = None
    category: str | None = None
    vendor: str | None = None
    price: float | None = None
    tile_w: float | None = None
    tile_h: float | None = None
    gloss: float | None = None


@app.put("/api/catalog/{sid}", dependencies=[api])
def catalog_update(sid: str, body: SKUUpdate):
    def work():
        entries = json.loads((C.ROOT / "catalog.json").read_text())
        e = next((e for e in entries if e["id"] == sid), None)
        if e is None:
            raise HTTPException(404, "no such SKU")
        e.update({k: v for k, v in body.model_dump().items() if v is not None})
        _validate_sku(e)
        write_catalog(entries)
        reload_catalog()
        return e
    return on_gpu(work)


@app.post("/api/catalog/{sid}/albedo", dependencies=[api])
def catalog_replace_texture(sid: str, albedo: UploadFile = File(...)):
    raw = albedo.file.read(MAX_UPLOAD + 1)

    def work():
        if sid not in engine.by_id:
            raise HTTPException(404, "no such SKU")
        _save_texture(sid, raw)
        reload_catalog()
    on_gpu(work)
    return {"ok": True}


@app.delete("/api/catalog/{sid}", dependencies=[api])
def catalog_delete(sid: str):
    def work():
        entries = json.loads((C.ROOT / "catalog.json").read_text())
        if not any(e["id"] == sid for e in entries):
            raise HTTPException(404, "no such SKU")
        write_catalog([e for e in entries if e["id"] != sid])
        if ID_RE.match(sid) and (C.ROOT / sid).is_dir():
            shutil.rmtree(C.ROOT / sid)
        reload_catalog()
    on_gpu(work)
    return {"ok": True}


# ------------------------------------------------------------------ projects
def _submit_analysis(pid: str, force: bool = False):
    """
    Queue an analysis and supervise it.

    The future used to be discarded, so an exception escaping _analyze_job's own
    handling (a corrupted project.json, an OOM kill inside a worker) vanished
    and the project sat at "analyzing" forever while the UI polled it.
    """
    future = gpu.submit(_analyze_job, pid, force)
    future.add_done_callback(lambda f: _analysis_done(pid, f))
    return future


def _analysis_done(pid: str, future):
    exc = future.exception()
    if exc is None:
        return
    log.error("analysis worker crashed for project %s", pid, exc_info=exc)

    def mark(meta):
        if meta.get("status") == "analyzing":
            meta.update(status="error", error=f"{type(exc).__name__}: {exc}")

    try:
        update_meta(pid, mark)
    except Exception:  # noqa: BLE001 - the project may have been deleted meanwhile
        log.warning("could not mark project %s as failed", pid)


def _analyze_job(pid: str, force: bool = False):
    """Runs on the GPU worker. Same photo bytes + same code/model version → copy the cached analysis
    instead of running detection and geometry again (force=True re-computes and refreshes the cache)."""
    try:
        t = time.time()
        meta = ensure_keys(pid, load_meta(pid))
        key = cache.analysis_key(meta["photo_sha"])
        hit = None if force else cache.get_analysis(key)
        if hit:
            npz, cached = hit
            shutil.copyfile(npz, pdir(pid) / "scene.tmp.npz")
            (pdir(pid) / "scene.tmp.npz").replace(pdir(pid) / "scene.npz")
            forget_scene(pid)                         # rebuilt lazily from disk when first needed
            fields = {k: cached[k] for k in ("notes", "timings", "surfaces", "size", "scene_rev")}
            fields["regions"] = cached.get("regions", [])   # entries cached before regions existed
            log.info("analysis cache hit for project %s", pid)
        else:
            sc = engine.analyze(Image.open(pdir(pid) / "photo.jpg"))
            rev = save_state(pid, sc)
            remember(pid, sc)
            fields = dict(notes=sc.notes, timings={k: round(v, 2) for k, v in sc.timings.items()},
                          surfaces=surfaces_summary(sc), regions=regions_summary(sc),
                          size=list(sc.image.size), scene_rev=rev)
            cache.put_analysis(key, pdir(pid) / "scene.npz", fields)
        # Mutate under the lock: a rename or a render delete may have landed
        # while the analysis was running, and must not be overwritten.
        def finish(m):
            m.update(status="ready", error=None, analysed_in=round(time.time() - t, 1),
                     analysis_cached=bool(hit),
                     photo_sha=m.get("photo_sha") or file_sha(pdir(pid) / "photo.jpg"),
                     corrections=[], **fields)

        update_meta(pid, finish)
    except HTTPException:
        return                  # project deleted while queued
    except Exception as e:  # noqa: BLE001
        log.exception("analysis failed for project %s", pid)
        try:
            update_meta(pid, lambda m: m.update(status="error", error=f"{type(e).__name__}: {e}"))
        except HTTPException:
            return


def _public(meta: dict) -> dict:
    return {k: v for k, v in meta.items() if k != "renders"} | dict(renders=meta.get("renders", [])[::-1])


@app.get("/api/projects", dependencies=[api])
def projects_list():
    out = []
    for d in sorted((DATA / "projects").iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            m = json.loads((d / "project.json").read_text())
            out.append(dict(id=m["id"], name=m["name"], status=m["status"], created=m["created"],
                            renders=len(m.get("renders", []))))
        except Exception:  # noqa: BLE001  (half-written / foreign folders)
            continue
    return out


@app.post("/api/projects", dependencies=[api])
def projects_create(photo: UploadFile = File(...), name: str = Form("")):
    im = read_image(photo.file.read(MAX_UPLOAD + 1))
    if min(im.size) < 256:
        raise HTTPException(400, "photo must be at least 256 px on the short side")
    pid = uuid.uuid4().hex[:12]
    d = DATA / "projects" / pid
    d.mkdir(parents=True)
    (d / "renders").mkdir()
    im.thumbnail((4096, 4096), Image.LANCZOS)
    im.save(d / "photo.jpg", quality=95)
    meta = dict(id=pid, name=(name or Path(photo.filename or "photo").stem)[:80], created=time.time(),
                status="analyzing", error=None, renders=[], assign={}, photo_sha=file_sha(d / "photo.jpg"))
    save_meta(pid, meta)
    _submit_analysis(pid)
    return _public(meta)


@app.get("/api/projects/{pid}", dependencies=[api])
def projects_get(pid: str):
    return _public(load_meta(pid))


@app.patch("/api/projects/{pid}", dependencies=[api])
def projects_rename(pid: str, body: dict):
    def rename(m):
        m["name"] = str(body.get("name", m["name"]))[:80] or m["name"]

    return _public(update_meta(pid, rename))


@app.delete("/api/projects/{pid}", dependencies=[api])
def projects_delete(pid: str):
    d = pdir(pid)
    def work():
        forget_scene(pid)
        shutil.rmtree(d)
    on_gpu(work)   # queue behind any running job on this project
    return {"ok": True}


@app.post("/api/projects/{pid}/reanalyze", dependencies=[api])
def projects_reanalyze(pid: str):
    meta = update_meta(pid, lambda m: m.update(status="analyzing", error=None))
    forget_scene(pid)
    _submit_analysis(pid, True)              # explicit re-analyse always recomputes
    return _public(meta)


@app.get("/api/projects/{pid}/photo", dependencies=[api])
def projects_photo(pid: str, full: bool = False, size: int | None = Query(None, ge=64, le=4096)):
    """full=0: the analysis-resolution image (tap coordinates are in these pixels)."""
    im, fu = Engine.prepare(Image.open(pdir(pid) / "photo.jpg"))
    return png(fu if full else im, size)


@app.get("/api/projects/{pid}/overlay", dependencies=[api])
def projects_overlay(pid: str):
    return png(on_gpu(lambda: overlay(get_scene(pid))))


class Correction(BaseModel):
    kind: str
    x: float = Field(ge=0)
    y: float = Field(ge=0)
    mode: str = "add"
    region: str | None = None   # which region to grow/shrink; None → all of `kind` (see edit_mask)


@app.post("/api/projects/{pid}/correct", dependencies=[api])
def projects_correct(pid: str, c: Correction):
    """Tap-to-correct. x, y are in analysis-image pixels (the /photo and /overlay size)."""
    if kind_of(c.kind) not in SURFACES or c.mode not in ("add", "remove"):
        raise HTTPException(400, "bad kind or mode")
    if c.region is not None and kind_of(c.region) not in SURFACES:
        raise HTTPException(400, "bad region")

    def work():
        sc = get_scene(pid)
        W, H = sc.image.size
        if c.x >= W or c.y >= H:
            raise HTTPException(400, f"point outside the {W}×{H} analysis image")
        engine.edit_mask(sc, c.kind, (c.x, c.y), add=c.mode == "add", region=c.region)
        rev = save_state(pid, sc)
        meta = load_meta(pid)
        meta["surfaces"] = surfaces_summary(sc)
        meta["regions"] = regions_summary(sc)
        meta["scene_rev"] = rev
        meta.setdefault("corrections", []).append(c.model_dump())
        save_meta(pid, meta)
        return _public(meta)
    return on_gpu(work)


class RenderReq(BaseModel):
    assign: dict[str, str | None]
    edges: bool = True
    reflections: bool = False
    grain: bool = True
    material_maps: bool = False   # SKU normal/roughness maps (experimental)
    local_tint: bool = False      # per-surface neutral-neighbour tint (experimental)
    harmonize: bool = False
    strength: float = Field(0.3, ge=0.1, le=0.5)


def _clean(o):
    """Report → JSON (numpy scalars, nested dicts)."""
    if isinstance(o, dict):
        return {str(k).rstrip("_"): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return round(float(o), 4)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


@app.post("/api/projects/{pid}/render", dependencies=[api])
def projects_render(pid: str, r: RenderReq):
    # "wall" assigns every wall; "wall#1" assigns one of them
    assign = {k: v for k, v in r.assign.items() if kind_of(k) in SURFACES and v}
    bad = [v for v in assign.values() if v not in engine.by_id]
    if bad:
        raise HTTPException(400, f"unknown SKU(s): {bad}")
    if not assign:
        raise HTTPException(400, "pick at least one material")

    options = r.model_dump(exclude={"assign"})

    def work():
        t = time.time()
        meta = load_meta(pid)
        if meta["status"] != "ready":
            raise HTTPException(409, f"project is {meta['status']}")
        if not meta.get("scene_rev") or not meta.get("photo_sha"):
            meta = ensure_keys(pid, meta)
            save_meta(pid, meta)
        # the image depends on the photo's pixels *and* the scene (analysis + corrections)
        key = cache.render_key(f"{meta['photo_sha']}:{meta['scene_rev']}", assign, options, engine.by_id)
        # 1) this project already has exactly this render → return it, no duplicate
        for old in reversed(meta["renders"]):
            if old.get("key") == key and (pdir(pid) / "renders" / f"{old['id']}.jpg").exists():
                meta["assign"] = assign
                save_meta(pid, meta)
                return old | dict(cached=True)
        rid = uuid.uuid4().hex[:10]
        dst = pdir(pid) / "renders" / f"{rid}.jpg"
        hit = cache.get_render(key)
        if hit:
            # 2) same scene + same style rendered before (any project) → copy, never share the file
            jpg, cached = hit
            shutil.copyfile(jpg, dst)
            rec = cached | dict(id=rid, created=time.time(), cached=True, seconds=round(time.time() - t, 2))
            log.info("render cache hit for project %s", pid)
        else:
            sc = get_scene(pid)
            out, bom, rep = engine.render(sc, assign, harmonize=r.harmonize, strength=r.strength,
                                          edges=r.edges, reflections=r.reflections, grain=r.grain,
                                          material_maps=r.material_maps, local_tint=r.local_tint)
            Image.fromarray((np.clip(out, 0, 1) * 255).astype(np.uint8)).save(dst, quality=93)
            rec = dict(id=rid, created=time.time(), key=key, assign=assign, options=options, bom=bom,
                       total=round(sum(b["est_cost"] for b in bom)), report=_clean(rep), cached=False,
                       seconds=round(time.time() - t, 2))
            # a harmonise request that fell back to the projection must not be served once weights exist
            harm_fell_back = r.harmonize and (engine._harm is None or "harmoniser failed" in str(rep.get("note")))
            if harm_fell_back:
                rec.pop("key")      # not reusable, not even within this project: recompute once SDXL works
            else:
                cache.put_render(key, dst, {k: v for k, v in rec.items() if k not in ("id", "created", "seconds")})
        def record(m):
            m.setdefault("renders", []).append(rec)
            m["assign"] = assign

        update_meta(pid, record)
        return rec
    return on_gpu(work)


@app.get("/api/cache", dependencies=[api])
def cache_stats():
    return on_gpu(cache.stats)


@app.delete("/api/cache", dependencies=[api])
def cache_clear():
    on_gpu(cache.clear)
    return {"ok": True}


@app.get("/api/projects/{pid}/renders/{rid}.jpg", dependencies=[api])
def projects_render_image(pid: str, rid: str, size: int | None = Query(None, ge=64, le=4096)):
    if not re.fullmatch(r"[a-f0-9]{10}", rid):
        raise HTTPException(404, "no such render")
    p = pdir(pid) / "renders" / f"{rid}.jpg"
    if not p.exists():
        raise HTTPException(404, "no such render")
    if size:
        return png(Image.open(p), size)
    return FileResponse(p, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@app.delete("/api/projects/{pid}/renders/{rid}", dependencies=[api])
def projects_render_delete(pid: str, rid: str):
    if not any(x["id"] == rid for x in load_meta(pid).get("renders", [])):
        raise HTTPException(404, "no such render")

    def drop(m):
        m["renders"] = [x for x in m.get("renders", []) if x["id"] != rid]

    update_meta(pid, drop)
    (pdir(pid) / "renders" / f"{rid}.jpg").unlink(missing_ok=True)
    return {"ok": True}


# ------------------------------------------------------------------ start-up + static UI
app.mount("/", StaticFiles(directory=Path(__file__).parent / "static", html=True), name="ui")
