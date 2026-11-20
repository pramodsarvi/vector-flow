"""
Spatial Scanner & Floor Designer — HTTP API.

A scan is a resource with a lifecycle: you create it (from an upload or the
bundled sample), the worker processes it in the background, and you poll it
until it is `ready`, then fetch its artifacts. Reconstruction never runs inside
a request, so the API stays responsive while a scan is being processed.

    POST   /api/scans                      create from an uploaded point cloud
    POST   /api/scans/sample               create from the bundled demo capture
    GET    /api/scans                      list scans, newest first
    GET    /api/scans/{id}                 one scan: status, progress, metrics
    POST   /api/scans/{id}/reprocess       re-run the pipeline on the same input
    DELETE /api/scans/{id}                 delete a scan and its artifacts
    GET    /api/scans/{id}/floorplan       the floorplan schema (JSON)
    GET    /api/scans/{id}/colliders       physics colliders (JSON)
    GET    /api/scans/{id}/artifacts/{name}  SVG / DXF / PDF / OBJ / GLB / PLY
    GET    /api/scans/{id}/frames/{file}   a captured RGB frame

`/api/scan/*` (singular) is kept as a thin compatibility layer over the most
recent ready scan, so the previous viewer and any existing scripts keep working.
"""

from __future__ import annotations

import hmac
import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from . import store, uploads
from .jobs import jobs
from .schemas import Config, Deleted, ErrorBody, Health, Scan, ScanList
from .settings import settings

VERSION = "2.0.0"

logging.basicConfig(
    level=getattr(logging, settings.log_level, logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    settings.scans_dir.mkdir(parents=True, exist_ok=True)
    jobs.start()
    log.info("spatial-scanner %s ready — data at %s", VERSION, settings.data_dir)
    # Re-queue anything the previous process was mid-way through, so a restart
    # does not leave a scan stuck at "processing" forever.
    for meta in store.listing():
        if meta.get("status") in (store.STATUS_QUEUED, store.STATUS_PROCESSING):
            log.info("re-queueing interrupted scan %s", meta["id"])
            jobs.submit(meta["id"])
    try:
        yield
    finally:
        jobs.stop()


app = FastAPI(
    title="Spatial Scanner & Floor Designer API",
    version=VERSION,
    description=__doc__,
    lifespan=lifespan,
    responses={
        400: {"model": ErrorBody}, 401: {"model": ErrorBody},
        404: {"model": ErrorBody}, 413: {"model": ErrorBody},
    },
)

# Same-origin by default: the viewer is served by this app, so no cross-origin
# access is needed unless an operator opts in with SCANNER_CORS_ORIGINS.
if settings.cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Authorization", "Content-Type"],
    )


# --------------------------------------------------------------------------
# Auth & error handling
# --------------------------------------------------------------------------

def require_auth(request: Request) -> None:
    """
    Shared-secret auth, enabled by setting SCANNER_TOKEN. Accepted as a bearer
    header or a `token` query parameter — the query form exists because <img>
    and <a download> cannot send headers.
    """
    if not settings.token:
        return
    header = request.headers.get("authorization", "")
    supplied = header[7:].strip() if header.lower().startswith("bearer ") else ""
    supplied = supplied or request.query_params.get("token", "")
    if not hmac.compare_digest(supplied, settings.token):
        raise HTTPException(status_code=401, detail="Missing or invalid API token.")


auth = Depends(require_auth)


def _error(status: int, code: str, detail: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": code, "detail": detail})


@app.exception_handler(store.ScanNotFound)
async def _not_found(_request: Request, exc: store.ScanNotFound) -> JSONResponse:
    return _error(404, "not_found", f"No such scan or artifact: {exc}")


@app.exception_handler(uploads.IngestError)
async def _bad_upload(_request: Request, exc: uploads.IngestError) -> JSONResponse:
    # 413 when the upload was simply too big, 400 when it was malformed.
    status = 413 if "limit" in str(exc).lower() else 400
    return _error(status, "invalid_upload", str(exc))


@app.exception_handler(HTTPException)
async def _http_error(_request: Request, exc: HTTPException) -> JSONResponse:
    codes = {400: "bad_request", 401: "unauthorized", 404: "not_found", 409: "conflict", 413: "too_large"}
    return _error(exc.status_code, codes.get(exc.status_code, "error"), str(exc.detail))


@app.exception_handler(Exception)
async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
    # Full detail to the log; the client gets a stable message with no internals.
    log.exception("unhandled error on %s %s", request.method, request.url.path)
    return _error(500, "internal_error", "The server hit an unexpected error. Check the service log.")


# --------------------------------------------------------------------------
# Service
# --------------------------------------------------------------------------

@app.get("/api/health", response_model=Health, dependencies=[auth], tags=["service"])
def health() -> Health:
    return Health(status="healthy", service="spatial-scanner-engine", version=VERSION, worker=jobs.status())


@app.get("/api/config", response_model=Config, tags=["service"])
def config() -> Config:
    """Public: the viewer needs this before it has a token."""
    return Config(**settings.public(), version=VERSION)


# --------------------------------------------------------------------------
# Scans
# --------------------------------------------------------------------------

def _to_scan(meta: dict) -> Scan:
    """Attach the artifact list, which is derived from disk rather than stored."""
    artifacts: list[str] = []
    if meta.get("status") == store.STATUS_READY:
        out = store.output_dir(meta["id"])
        artifacts = [name for name in store.ARTIFACTS if (out / name).exists()]
    return Scan(**meta, artifacts=artifacts)


@app.get("/api/scans", response_model=ScanList, dependencies=[auth], tags=["scans"])
def scans_list(limit: int = 100, offset: int = 0) -> ScanList:
    all_scans = store.listing()
    limit = max(1, min(limit, 500))
    page = all_scans[offset: offset + limit]
    return ScanList(scans=[_to_scan(m) for m in page], total=len(all_scans))


@app.post("/api/scans", response_model=Scan, status_code=202, dependencies=[auth], tags=["scans"])
async def scans_create(
    cloud: UploadFile = File(..., description="LiDAR point cloud (.ply, ASCII or binary)"),
    poses: UploadFile | None = File(None, description="Optional camera_poses.json"),
    frames: list[UploadFile] = File([], description="Optional captured RGB frames"),
    name: str = Form("", description="Display name; defaults to the scan id"),
) -> Scan:
    """
    Create a scan from an uploaded capture bundle and queue it for processing.

    Responds 202 immediately; poll GET /api/scans/{id} for progress.
    """
    meta = store.create(name=name, source="upload")
    scan_id = meta["id"]
    input_dir = store.input_dir(scan_id)
    try:
        uploads.accept_point_cloud(cloud.file, input_dir)
        if poses is not None and poses.filename:
            uploads.accept_poses(poses.file, input_dir)
        for frame in frames:
            if frame.filename:
                uploads.accept_frame(frame.file, input_dir, frame.filename)
    except Exception:
        # Never leave a half-ingested scan in the store.
        store.delete(scan_id)
        raise

    jobs.submit(scan_id)
    return _to_scan(store.read_meta(scan_id))


@app.post("/api/scans/sample", response_model=Scan, status_code=202, dependencies=[auth], tags=["scans"])
def scans_create_sample(name: str = Form("Sample room scan")) -> Scan:
    """Create a scan from the bundled demo capture — the fastest way to see the product work."""
    meta = store.create(name=name, source="sample")
    scan_id = meta["id"]
    try:
        uploads.copy_sample_bundle(store.input_dir(scan_id))
    except Exception:
        store.delete(scan_id)
        raise
    jobs.submit(scan_id)
    return _to_scan(store.read_meta(scan_id))


@app.get("/api/scans/{scan_id}", response_model=Scan, dependencies=[auth], tags=["scans"])
def scans_get(scan_id: str) -> Scan:
    return _to_scan(store.read_meta(scan_id))


@app.post("/api/scans/{scan_id}/reprocess", response_model=Scan, status_code=202,
          dependencies=[auth], tags=["scans"])
def scans_reprocess(scan_id: str) -> Scan:
    """Re-run the pipeline over the scan's existing input bundle."""
    meta = store.read_meta(scan_id)
    if meta["status"] in (store.STATUS_QUEUED, store.STATUS_PROCESSING):
        raise HTTPException(status_code=409, detail="This scan is already being processed.")
    if not (store.input_dir(scan_id) / "raw_lidar_cloud.ply").is_file():
        raise HTTPException(status_code=409, detail="This scan's input point cloud is no longer on disk.")
    jobs.submit(scan_id)
    return _to_scan(store.read_meta(scan_id))


@app.delete("/api/scans/{scan_id}", response_model=Deleted, dependencies=[auth], tags=["scans"])
def scans_delete(scan_id: str) -> Deleted:
    store.delete(scan_id)
    return Deleted(deleted=scan_id)


# --------------------------------------------------------------------------
# Artifacts
# --------------------------------------------------------------------------

def _require_ready(scan_id: str) -> dict:
    meta = store.read_meta(scan_id)
    if meta["status"] != store.STATUS_READY:
        raise HTTPException(
            status_code=409,
            detail=f"Scan is '{meta['status']}', not ready. Artifacts appear once processing succeeds.",
        )
    return meta


@app.get("/api/scans/{scan_id}/floorplan", dependencies=[auth], tags=["artifacts"])
def scan_floorplan(scan_id: str) -> FileResponse:
    """The floorplan schema: room polygon, walls, openings, furniture, capture metadata."""
    _require_ready(scan_id)
    return FileResponse(store.artifact_path(scan_id, "floorplan.json"), media_type="application/json")


@app.get("/api/scans/{scan_id}/colliders", dependencies=[auth], tags=["artifacts"])
def scan_colliders(scan_id: str) -> FileResponse:
    _require_ready(scan_id)
    return FileResponse(store.artifact_path(scan_id, "colliders.json"), media_type="application/json")


@app.get("/api/scans/{scan_id}/artifacts/{name}", dependencies=[auth], tags=["artifacts"])
def scan_artifact(scan_id: str, name: str) -> FileResponse:
    """Download an artifact. `name` is matched against an allowlist, never used as a path."""
    _require_ready(scan_id)
    path = store.artifact_path(scan_id, name)
    media_type, as_download = store.ARTIFACTS[name]
    return FileResponse(
        path,
        media_type=media_type,
        filename=f"{scan_id}-{name}" if as_download else None,
    )


@app.get("/api/scans/{scan_id}/frames/{filename}", dependencies=[auth], tags=["artifacts"])
def scan_frame(scan_id: str, filename: str) -> FileResponse:
    """A captured RGB frame, as referenced by `scan.camera_views[].image_file`."""
    path = store.frame_path(scan_id, filename)
    suffix = path.suffix.lower()
    media_type = "image/png" if suffix == ".png" else "image/jpeg"
    return FileResponse(path, media_type=media_type)


# --------------------------------------------------------------------------
# Compatibility: /api/scan/* serves the most recent ready scan
# --------------------------------------------------------------------------

def _latest_or_404() -> dict:
    meta = store.latest_ready()
    if meta is None:
        raise HTTPException(status_code=404, detail="No processed scan yet. Create one via POST /api/scans.")
    return meta


@app.get("/api/scan/latest", response_model=Scan, dependencies=[auth], tags=["compatibility"])
def scan_latest() -> Scan:
    return _to_scan(_latest_or_404())


@app.post("/api/scan/process", response_model=Scan, status_code=202,
          dependencies=[auth], tags=["compatibility"])
def scan_process() -> Scan:
    """Legacy trigger: processes the bundled sample as a new scan."""
    return scans_create_sample()


@app.post("/api/scan/upload", response_model=Scan, status_code=202,
          dependencies=[auth], tags=["compatibility"])
async def scan_upload(file: UploadFile = File(...)) -> Scan:
    """Legacy upload endpoint, kept so existing clients keep working."""
    return await scans_create(cloud=file, poses=None, frames=[], name="")


_LEGACY_ARTIFACTS = {
    "floorplan": "floorplan.json",
    "svg": "floorplan.svg",
    "dxf": "floorplan.dxf",
    "pdf": "floorplan.pdf",
    "colliders": "colliders.json",
    "pointcloud": "aligned_point_cloud.ply",
    "model/glb": "room_model.glb",
    "model/obj": "room_model.obj",
}


@app.get("/api/scan/{legacy:path}", dependencies=[auth], tags=["compatibility"])
def scan_legacy_artifact(legacy: str) -> Response:
    name = _LEGACY_ARTIFACTS.get(legacy.strip("/"))
    if name is None:
        raise HTTPException(status_code=404, detail=f"Unknown artifact '{legacy}'.")
    meta = _latest_or_404()
    return RedirectResponse(url=f"/api/scans/{meta['id']}/artifacts/{name}", status_code=307)


# --------------------------------------------------------------------------
# Static viewer
# --------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse(url="/viewer/")


if settings.viewer_dir.is_dir():
    app.mount("/viewer", StaticFiles(directory=str(settings.viewer_dir), html=True), name="viewer")
else:  # pragma: no cover - only when deployed API-only
    log.warning("viewer directory not found at %s; serving API only", settings.viewer_dir)
