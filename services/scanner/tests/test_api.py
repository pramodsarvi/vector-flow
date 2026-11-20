"""
The HTTP service: scan lifecycle, artifact serving, validation and auth.

These run the real pipeline on the real sample bundle — the service is only
useful if the whole path works, and the sample takes under two seconds.
"""

import io
import json

import numpy as np
import pytest

from engine.ingest import ply_io


def _cloud_bytes(n=4000, binary=False, seed=0):
    """A synthetic room cloud, as a PLY upload would arrive."""
    rng = np.random.default_rng(seed)
    per = n // 6
    length, width, height = 5.0, 4.0, 2.7
    parts = [
        np.column_stack([rng.uniform(0, length, per), np.zeros(per), rng.uniform(0, width, per)]),
        np.column_stack([rng.uniform(0, length, per), np.full(per, height), rng.uniform(0, width, per)]),
        np.column_stack([rng.uniform(0, length, per), rng.uniform(0, height, per), np.zeros(per)]),
        np.column_stack([rng.uniform(0, length, per), rng.uniform(0, height, per), np.full(per, width)]),
        np.column_stack([np.zeros(per), rng.uniform(0, height, per), rng.uniform(0, width, per)]),
        np.column_stack([np.full(per, length), rng.uniform(0, height, per), rng.uniform(0, width, per)]),
    ]
    points = np.vstack(parts).astype(np.float32)

    buffer = io.BytesIO()
    import os
    import tempfile

    handle, path = tempfile.mkstemp(suffix=".ply")
    os.close(handle)
    try:
        ply_io.write_ply(path, points, binary=binary)
        with open(path, "rb") as f:
            buffer.write(f.read())
    finally:
        os.unlink(path)
    return buffer.getvalue()


class TestService:
    def test_config_is_public(self, client):
        body = client.get("/api/config").json()
        assert body["auth_required"] is False
        assert body["sample_available"] is True
        assert body["max_upload_mb"] > 0

    def test_health_reports_the_worker(self, client):
        body = client.get("/api/health").json()
        assert body["status"] == "healthy"
        assert "worker" in body

    def test_root_redirects_to_the_viewer(self, client):
        response = client.get("/", follow_redirects=False)
        assert response.status_code in (307, 302)
        assert "/viewer" in response.headers["location"]

    def test_openapi_documents_the_api(self, client):
        schema = client.get("/openapi.json").json()
        assert "/api/scans" in schema["paths"]


class TestScanLifecycle:
    def test_sample_scan_processes_and_exposes_artifacts(self, client, wait_for_scan):
        created = client.post("/api/scans/sample")
        assert created.status_code == 202
        scan_id = created.json()["id"]
        assert created.json()["status"] == "queued"

        done = wait_for_scan(client, scan_id)
        assert done["status"] == "ready", done.get("error")
        assert done["duration_s"] > 0
        assert done["progress"]["step"] == 6

        summary = done["summary"]
        assert summary["area_sqm"] == pytest.approx(27.0, abs=0.2)
        assert summary["door_count"] == 1
        assert summary["window_count"] == 1
        assert summary["furniture_count"] == 2
        assert summary["camera_view_count"] == 12

        for name in done["artifacts"]:
            response = client.get(f"/api/scans/{scan_id}/artifacts/{name}")
            assert response.status_code == 200, name
            assert len(response.content) > 0

    def test_floorplan_and_colliders_endpoints(self, client, wait_for_scan):
        scan_id = client.post("/api/scans/sample").json()["id"]
        wait_for_scan(client, scan_id)

        floorplan = client.get(f"/api/scans/{scan_id}/floorplan").json()
        assert floorplan["units"] == "meters"
        assert len(floorplan["room"]["walls"]) == 4

        colliders = client.get(f"/api/scans/{scan_id}/colliders").json()
        assert any(c["type"] == "wall_segment" for c in colliders["colliders"])

    def test_serves_capture_frames(self, client, wait_for_scan):
        scan_id = client.post("/api/scans/sample").json()["id"]
        wait_for_scan(client, scan_id)
        floorplan = client.get(f"/api/scans/{scan_id}/floorplan").json()
        image_file = floorplan["scan"]["camera_views"][0]["image_file"]

        response = client.get(f"/api/scans/{scan_id}/frames/{image_file.split('/')[-1]}")
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/jpeg"

    def test_list_is_newest_first(self, client, wait_for_scan):
        first = client.post("/api/scans/sample", data={"name": "first"}).json()["id"]
        wait_for_scan(client, first)
        second = client.post("/api/scans/sample", data={"name": "second"}).json()["id"]
        wait_for_scan(client, second)

        body = client.get("/api/scans").json()
        assert body["total"] == 2
        assert body["scans"][0]["id"] == second

    def test_reprocess_reruns_the_pipeline(self, client, wait_for_scan):
        scan_id = client.post("/api/scans/sample").json()["id"]
        wait_for_scan(client, scan_id)

        again = client.post(f"/api/scans/{scan_id}/reprocess")
        assert again.status_code == 202
        assert wait_for_scan(client, scan_id)["status"] == "ready"

    def test_delete_removes_the_scan_and_its_files(self, client, wait_for_scan, data_dir):
        scan_id = client.post("/api/scans/sample").json()["id"]
        wait_for_scan(client, scan_id)
        assert (data_dir / "scans" / scan_id).exists()

        assert client.delete(f"/api/scans/{scan_id}").status_code == 200
        assert not (data_dir / "scans" / scan_id).exists()
        assert client.get(f"/api/scans/{scan_id}").status_code == 404

    def test_upload_is_processed(self, client, wait_for_scan):
        response = client.post(
            "/api/scans",
            files={"cloud": ("room.ply", _cloud_bytes(), "application/octet-stream")},
            data={"name": "uploaded room"},
        )
        assert response.status_code == 202
        done = wait_for_scan(client, response.json()["id"])
        assert done["status"] == "ready", done.get("error")
        assert done["name"] == "uploaded room"
        assert done["summary"]["area_sqm"] == pytest.approx(20.0, abs=1.5)

    def test_binary_upload_is_accepted(self, client, wait_for_scan):
        response = client.post(
            "/api/scans",
            files={"cloud": ("room.ply", _cloud_bytes(binary=True), "application/octet-stream")},
        )
        assert response.status_code == 202
        assert wait_for_scan(client, response.json()["id"])["status"] == "ready"

    def test_upload_with_poses_and_frames(self, client, wait_for_scan):
        poses = json.dumps({"frames": [
            {"frame_id": 0, "image_file": "images/frame_0000.jpg", "camera_position": [1.0, 1.4, 1.0]},
        ]}).encode()
        response = client.post(
            "/api/scans",
            files=[
                ("cloud", ("room.ply", _cloud_bytes(), "application/octet-stream")),
                ("poses", ("camera_poses.json", poses, "application/json")),
                ("frames", ("frame_0000.jpg", b"\xff\xd8\xff\xe0 fake jpeg", "image/jpeg")),
            ],
        )
        assert response.status_code == 202
        done = wait_for_scan(client, response.json()["id"])
        assert done["status"] == "ready", done.get("error")
        assert done["summary"]["camera_view_count"] == 1


class TestValidation:
    def test_rejects_a_file_that_is_not_a_point_cloud(self, client):
        response = client.post(
            "/api/scans",
            files={"cloud": ("evil.ply", b"definitely not a ply", "application/octet-stream")},
        )
        assert response.status_code == 400
        assert "PLY" in response.json()["detail"]

    def test_rejected_upload_leaves_no_scan_behind(self, client):
        client.post("/api/scans", files={"cloud": ("x.ply", b"nope", "application/octet-stream")})
        assert client.get("/api/scans").json()["total"] == 0

    def test_rejects_an_empty_upload(self, client):
        response = client.post("/api/scans", files={"cloud": ("x.ply", b"", "application/octet-stream")})
        assert response.status_code == 400

    def test_rejects_oversized_uploads(self, client, monkeypatch):
        from engine.service import settings, uploads

        monkeypatch.setattr(type(settings.settings), "max_upload_bytes", property(lambda self: 1024))
        response = client.post(
            "/api/scans",
            files={"cloud": ("big.ply", _cloud_bytes(n=6000), "application/octet-stream")},
        )
        assert response.status_code == 413
        assert "limit" in response.json()["detail"].lower()
        assert uploads  # imported for clarity about what is under test

    def test_rejects_malformed_poses(self, client):
        response = client.post(
            "/api/scans",
            files=[
                ("cloud", ("room.ply", _cloud_bytes(), "application/octet-stream")),
                ("poses", ("camera_poses.json", b"{not json", "application/json")),
            ],
        )
        assert response.status_code == 400
        assert "JSON" in response.json()["detail"]

    def test_rejects_a_frame_with_a_dangerous_name(self, client):
        response = client.post(
            "/api/scans",
            files=[
                ("cloud", ("room.ply", _cloud_bytes(), "application/octet-stream")),
                ("frames", ("../../etc/passwd", b"x", "image/jpeg")),
            ],
        )
        assert response.status_code == 400

    @pytest.mark.parametrize("scan_id", [
        "../../etc", "not-an-id", "20261117-120000-zzzzzz", "....//....//etc",
    ])
    def test_rejects_malformed_scan_ids(self, client, scan_id):
        assert client.get(f"/api/scans/{scan_id}").status_code == 404

    def test_artifact_names_are_allowlisted(self, client, wait_for_scan):
        scan_id = client.post("/api/scans/sample").json()["id"]
        wait_for_scan(client, scan_id)
        for name in ("../scan.json", "scan.json", "..%2fscan.json", "secrets.txt"):
            assert client.get(f"/api/scans/{scan_id}/artifacts/{name}").status_code == 404

    def test_frame_paths_cannot_escape_the_scan(self, client, wait_for_scan):
        scan_id = client.post("/api/scans/sample").json()["id"]
        wait_for_scan(client, scan_id)
        for name in ("../../../etc/passwd", "../scan.json", "frame_0000.txt"):
            assert client.get(f"/api/scans/{scan_id}/frames/{name}").status_code == 404

    def test_artifacts_404_until_the_scan_is_ready(self, client):
        scan_id = client.post("/api/scans/sample").json()["id"]
        response = client.get(f"/api/scans/{scan_id}/floorplan")
        # Either still queued (409) or already finished by the time we asked.
        assert response.status_code in (200, 409)

    def test_errors_have_a_consistent_shape(self, client):
        body = client.get("/api/scans/not-an-id").json()
        assert set(body) == {"error", "detail"}
        assert isinstance(body["detail"], str)


class TestCompatibility:
    def test_legacy_routes_follow_the_latest_scan(self, client, wait_for_scan):
        scan_id = client.post("/api/scans/sample").json()["id"]
        wait_for_scan(client, scan_id)

        assert client.get("/api/scan/latest").json()["id"] == scan_id
        for legacy in ("floorplan", "svg", "dxf", "pdf", "colliders", "pointcloud", "model/glb"):
            response = client.get(f"/api/scan/{legacy}")
            assert response.status_code == 200, legacy

    def test_legacy_routes_404_without_a_scan(self, client):
        assert client.get("/api/scan/floorplan").status_code == 404

    def test_legacy_upload_still_works(self, client, wait_for_scan):
        response = client.post(
            "/api/scan/upload",
            files={"file": ("room.ply", _cloud_bytes(), "application/octet-stream")},
        )
        assert response.status_code == 202
        assert wait_for_scan(client, response.json()["id"])["status"] == "ready"


class TestAuth:
    @pytest.fixture
    def secure_client(self, tmp_path, monkeypatch):
        from fastapi.testclient import TestClient

        monkeypatch.setenv("SCANNER_DATA", str(tmp_path / "secure"))
        monkeypatch.setenv("SCANNER_TOKEN", "s3cret")
        monkeypatch.setenv("SCANNER_LOG", "WARNING")
        from tests.conftest import _reload_service

        _reload_service()
        from engine.service.app import app

        with TestClient(app) as client:
            yield client

    def test_config_stays_public(self, secure_client):
        body = secure_client.get("/api/config").json()
        assert body["auth_required"] is True

    def test_requests_without_a_token_are_rejected(self, secure_client):
        assert secure_client.get("/api/scans").status_code == 401
        assert secure_client.get("/api/health").status_code == 401

    def test_bearer_header_is_accepted(self, secure_client):
        response = secure_client.get("/api/scans", headers={"Authorization": "Bearer s3cret"})
        assert response.status_code == 200

    def test_query_token_is_accepted_for_downloads(self, secure_client):
        assert secure_client.get("/api/scans?token=s3cret").status_code == 200

    def test_a_wrong_token_is_rejected(self, secure_client):
        assert secure_client.get("/api/scans", headers={"Authorization": "Bearer nope"}).status_code == 401
