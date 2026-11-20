"""
Single-worker background job queue.

Reconstruction takes seconds to minutes and is CPU-bound, so it must not run
inside a request handler — a blocked worker makes the whole API unresponsive.
Exactly one job runs at a time on purpose: the pipeline seeds the global numpy
RNG for reproducible RANSAC, which is not safe to run concurrently.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
import traceback
from collections.abc import Callable

from . import store

log = logging.getLogger(__name__)

# Injected so tests can run a fake pipeline without importing the CV stack.
PipelineFn = Callable[..., dict]


class JobQueue:
    """FIFO queue drained by one daemon thread."""

    def __init__(self, pipeline: PipelineFn | None = None) -> None:
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._current: str | None = None
        self._lock = threading.Lock()
        self._pipeline = pipeline

    # ---- lifecycle -------------------------------------------------------

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run, name="scan-worker", daemon=True)
        self._thread.start()
        log.info("scan worker started")

    def stop(self, timeout: float = 10.0) -> None:
        """Ask the worker to finish the current job and exit."""
        if not (self._thread and self._thread.is_alive()):
            return
        self._queue.put(None)
        self._thread.join(timeout=timeout)
        log.info("scan worker stopped")

    # ---- submission ------------------------------------------------------

    def submit(self, scan_id: str) -> None:
        store.update(
            scan_id,
            status=store.STATUS_QUEUED,
            error=None,
            progress={"step": 0, "total": 6, "message": "Queued"},
        )
        self._queue.put(scan_id)

    def status(self) -> dict:
        with self._lock:
            current = self._current
        return {"running": current, "queued": max(0, self._queue.qsize() - (0 if current else 0))}

    # ---- worker ----------------------------------------------------------

    def _resolve_pipeline(self) -> PipelineFn:
        if self._pipeline is None:
            # Imported lazily: the heavy CV dependencies load on first job, not
            # at import time, so the API starts fast and tests stay light.
            from ..pipeline import process_spatial_scan

            self._pipeline = process_spatial_scan
        return self._pipeline

    def _run(self) -> None:
        while True:
            scan_id = self._queue.get()
            if scan_id is None:
                self._queue.task_done()
                return
            with self._lock:
                self._current = scan_id
            try:
                self._process(scan_id)
            except Exception:  # pragma: no cover - last-resort guard
                log.exception("worker crashed on scan %s", scan_id)
            finally:
                with self._lock:
                    self._current = None
                self._queue.task_done()

    def _process(self, scan_id: str) -> None:
        started = time.monotonic()
        try:
            store.update(
                scan_id,
                status=store.STATUS_PROCESSING,
                progress={"step": 0, "total": 6, "message": "Starting"},
            )
        except store.ScanNotFound:
            log.warning("scan %s vanished before processing", scan_id)
            return

        def on_progress(step: int, total: int, message: str) -> None:
            try:
                store.set_progress(scan_id, step, total, message)
            except store.ScanNotFound:  # deleted mid-run; let the pipeline finish harmlessly
                pass

        try:
            floorplan = self._resolve_pipeline()(
                str(store.input_dir(scan_id)),
                str(store.output_dir(scan_id)),
                progress=on_progress,
            )
        except store.ScanNotFound:
            return
        except Exception as exc:
            # Full detail to the log, a short safe message to the client.
            log.error("scan %s failed: %s\n%s", scan_id, exc, traceback.format_exc())
            try:
                store.update(
                    scan_id,
                    status=store.STATUS_FAILED,
                    error=_safe_error(exc),
                    duration_s=round(time.monotonic() - started, 2),
                    progress={"step": 0, "total": 6, "message": "Failed"},
                )
            except store.ScanNotFound:
                pass
            return

        try:
            store.update(
                scan_id,
                status=store.STATUS_READY,
                summary=store.summarize(floorplan),
                duration_s=round(time.monotonic() - started, 2),
                progress={"step": 6, "total": 6, "message": "Complete"},
                error=None,
            )
        except store.ScanNotFound:
            return
        store.prune()
        log.info("scan %s ready in %.2fs", scan_id, time.monotonic() - started)


def _safe_error(exc: Exception) -> str:
    """
    A one-line, client-safe description. Exception text can carry absolute
    paths and library internals, so only the type and a trimmed message go out.
    """
    message = str(exc).strip().splitlines()[0] if str(exc).strip() else exc.__class__.__name__
    if len(message) > 200:
        message = message[:197] + "..."
    return f"{exc.__class__.__name__}: {message}"


# The app-wide queue. Started and stopped by the FastAPI lifespan.
jobs = JobQueue()
