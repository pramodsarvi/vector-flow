"""
End-to-end pipeline on the bundled sample scan.

These assertions are the product's accuracy claims. The sample room is
generated with known dimensions (6.00 m x 4.50 m, a 0.90 m door and a 2.00 m
window), so they are checked against ground truth, not against whatever the
code happens to produce.
"""

import json

import pytest

from engine.ingest import ScanInputError
from engine.pipeline import process_spatial_scan

GROUND_TRUTH = {
    "length_m": 6.0,
    "width_m": 4.5,
    "area_sqm": 27.0,
    "ceiling_height_m": 2.8,
    "door_width_m": 0.90,
    "window_width_m": 2.00,
}


@pytest.fixture(scope="module")
def result(tmp_path_factory, request):
    sample = request.config.rootpath / "capture" / "sample-data"
    if not (sample / "raw_lidar_cloud.ply").is_file():
        pytest.skip("sample scan bundle is not installed")
    out = tmp_path_factory.mktemp("pipeline-out")
    floorplan = process_spatial_scan(str(sample), str(out))
    return floorplan, out


class TestArtifacts:
    def test_writes_every_deliverable(self, result):
        _, out = result
        for name in (
            "floorplan.json", "floorplan.svg", "floorplan.dxf", "floorplan.pdf",
            "room_model.obj", "room_model.glb", "colliders.json", "aligned_point_cloud.ply",
        ):
            path = out / name
            assert path.is_file(), f"{name} was not written"
            assert path.stat().st_size > 0, f"{name} is empty"

    def test_json_on_disk_matches_the_returned_schema(self, result):
        floorplan, out = result
        on_disk = json.loads((out / "floorplan.json").read_text())
        assert on_disk["room"] == floorplan["room"]

    def test_prepares_the_splatting_dataset(self, result):
        _, out = result
        transforms = out / "3dgs" / "transforms.json"
        assert transforms.is_file()
        data = json.loads(transforms.read_text())
        assert len(data["frames"]) == 12

    def test_does_not_touch_the_viewer_unless_asked(self, result, request):
        """
        The viewer reads scans from the API. The pipeline writing copies next
        to it is what previously let the two diverge, so it is opt-in now.
        """
        _, out = result
        viewer = request.config.rootpath / "viewer"
        assert not (viewer / "floorplan.json").exists()


class TestMeasurements:
    def test_room_dimensions_match_ground_truth(self, result):
        floorplan, _ = result
        lengths = sorted(w["length_m"] for w in floorplan["room"]["walls"])
        assert lengths[0] == pytest.approx(GROUND_TRUTH["width_m"], abs=0.05)
        assert lengths[-1] == pytest.approx(GROUND_TRUTH["length_m"], abs=0.05)

    def test_floor_area_is_within_half_a_percent(self, result):
        floorplan, _ = result
        area = floorplan["room"]["area_sqm"]
        error = abs(area - GROUND_TRUTH["area_sqm"]) / GROUND_TRUTH["area_sqm"]
        assert error < 0.005, f"area {area} m² is {error:.2%} off the 27.00 m² ground truth"

    def test_detects_one_door_and_one_window_at_the_right_widths(self, result):
        floorplan, _ = result
        openings = floorplan["room"]["openings"]
        doors = [o for o in openings if o["type"] == "door"]
        windows = [o for o in openings if o["type"] == "window"]

        assert len(doors) == 1
        assert len(windows) == 1
        assert doors[0]["width_m"] == pytest.approx(GROUND_TRUTH["door_width_m"], abs=0.11)
        assert windows[0]["width_m"] == pytest.approx(GROUND_TRUTH["window_width_m"], abs=0.11)
        assert windows[0]["sill_height_m"] > 0.5, "a window should sit above the floor"
        assert doors[0]["sill_height_m"] == pytest.approx(0.0, abs=0.01)

    def test_detects_both_furniture_pieces(self, result):
        floorplan, _ = result
        furniture = floorplan["room"]["furniture_footprints"]
        assert len(furniture) == 2
        for item in furniture:
            assert item["bbox"]["width_m"] > 0.3
            assert item["bbox"]["height_m"] > 0.2

    def test_schema_is_self_consistent(self, result):
        floorplan, _ = result
        room = floorplan["room"]
        assert floorplan["units"] == "meters"
        assert len(room["walls"]) == len(room["polygon"])
        # Perimeter must equal the sum of the wall lengths it is derived from.
        assert room["perimeter_m"] == pytest.approx(sum(w["length_m"] for w in room["walls"]), abs=0.02)
        assert room["area_sqft"] == pytest.approx(room["area_sqm"] * 10.7639, rel=0.01)

    def test_camera_views_are_in_the_aligned_frame(self, result):
        floorplan, _ = result
        views = floorplan["scan"]["camera_views"]
        assert len(views) == 12
        for view in views:
            x, y, z = view["position"]
            assert -2 < x < 9 and -2 < z < 8, "camera should be near the room"
            assert 0 < y < 4, "camera should be above the floor and below the ceiling"


class TestColliders:
    def test_walls_are_split_around_the_door(self, result):
        """
        The door gap must be absent from the collider wall segments, otherwise
        the walkthrough cannot pass through a doorway.
        """
        _, out = result
        colliders = json.loads((out / "colliders.json").read_text())["colliders"]
        segments = [c for c in colliders if c["type"] == "wall_segment"]
        # A plain rectangular room has 4 walls; one is split in two by the door.
        assert len(segments) == 5
        assert any(c["type"] == "plane" for c in colliders)
        assert len([c for c in colliders if c["type"] == "box"]) == 2


class TestFailures:
    def test_missing_directory(self, tmp_path):
        with pytest.raises(ScanInputError, match="does not exist"):
            process_spatial_scan(str(tmp_path / "nope"), str(tmp_path / "out"))

    def test_missing_point_cloud(self, tmp_path):
        (tmp_path / "empty").mkdir()
        with pytest.raises(ScanInputError, match="No point cloud"):
            process_spatial_scan(str(tmp_path / "empty"), str(tmp_path / "out"))

    def test_reports_progress_for_every_stage(self, sample_dir, tmp_path):
        seen = []
        process_spatial_scan(
            str(sample_dir), str(tmp_path / "out"),
            progress=lambda step, total, message: seen.append((step, total, message)),
        )
        assert [s[0] for s in seen] == [1, 2, 3, 4, 5, 6]
        assert all(s[1] == 6 for s in seen)
        assert all(s[2] for s in seen), "every stage should carry a message"
