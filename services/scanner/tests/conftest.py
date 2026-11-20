"""
Shared fixtures.

Every test runs against an isolated data directory, so a test run never
touches a developer's real scans and tests cannot leak state into each other.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SAMPLE = ROOT / "capture" / "sample-data"

# `engine` is an ordinary package under the repo root; one entry is enough.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def sample_dir() -> Path:
    if not (SAMPLE / "raw_lidar_cloud.ply").is_file():
        pytest.skip("sample scan bundle is not installed")
    return SAMPLE


@pytest.fixture
def data_dir(tmp_path, monkeypatch) -> Path:
    """Point the service at a throwaway data directory for this test."""
    target = tmp_path / "data"
    monkeypatch.setenv("SCANNER_DATA", str(target))
    monkeypatch.setenv("SCANNER_LOG", "WARNING")
    _reload_service()
    return target


@pytest.fixture
def client(data_dir):
    """A TestClient with the background worker running, torn down afterwards."""
    from fastapi.testclient import TestClient

    from engine.service.app import app

    with TestClient(app) as test_client:
        yield test_client


def _reload_service():
    """
    Re-import the service modules so they pick up the patched environment.

    Settings are read once at import time (deliberately — the service does not
    re-read the environment per request), so a test that changes them has to
    reload the modules that captured them.
    """
    import importlib

    for name in ("engine.service.settings", "engine.service.store", "engine.service.uploads",
                 "engine.service.jobs", "engine.service.schemas", "engine.service.app"):
        module = sys.modules.get(name)
        if module is not None:
            importlib.reload(module)


@pytest.fixture
def wait_for_scan():
    """Block until a scan leaves the queued/processing states."""
    def _wait(test_client, scan_id, timeout=180.0):
        import time

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            body = test_client.get(f"/api/scans/{scan_id}").json()
            if body["status"] in ("ready", "failed"):
                return body
            time.sleep(0.05)
        raise AssertionError(f"scan {scan_id} did not finish within {timeout}s")

    return _wait
