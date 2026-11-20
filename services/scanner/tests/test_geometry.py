"""
Alignment and vectorization: the guards that keep a bad scan from producing
silent nonsense, plus the measurements the product is sold on.
"""

import numpy as np
import pytest

from engine.geometry import alignment, vectorize


def _box_room(length=6.0, width=4.5, height=2.8, n=14000, seed=0):
    """A synthetic room: floor, ceiling and four walls with a little noise."""
    rng = np.random.default_rng(seed)
    per = n // 6
    floor = np.column_stack([rng.uniform(0, length, per), np.zeros(per), rng.uniform(0, width, per)])
    ceiling = np.column_stack([rng.uniform(0, length, per), np.full(per, height), rng.uniform(0, width, per)])
    walls = [
        np.column_stack([rng.uniform(0, length, per), rng.uniform(0, height, per), np.zeros(per)]),
        np.column_stack([rng.uniform(0, length, per), rng.uniform(0, height, per), np.full(per, width)]),
        np.column_stack([np.zeros(per), rng.uniform(0, height, per), rng.uniform(0, width, per)]),
        np.column_stack([np.full(per, length), rng.uniform(0, height, per), rng.uniform(0, width, per)]),
    ]
    points = np.vstack([floor, ceiling, *walls])
    return points + rng.normal(0, 0.004, points.shape)


class TestAlignment:
    def test_levels_a_tilted_room(self):
        points = _box_room()
        # Tilt the whole capture 12 degrees, as a hand-held rig would.
        angle = np.radians(12)
        tilt = np.array([
            [1, 0, 0],
            [0, np.cos(angle), -np.sin(angle)],
            [0, np.sin(angle), np.cos(angle)],
        ])
        aligned, transform, plane = alignment.align_floor_to_y_up(points @ tilt.T)

        floor_band = aligned[np.abs(aligned[:, 1]) < 0.05]
        assert len(floor_band) > 1000, "the floor should sit at Y=0 after alignment"
        assert transform.shape == (4, 4)
        assert len(plane) == 4

    def test_gravity_prior_picks_the_floor_not_a_wall(self):
        """A room whose largest plane is a wall must still align on the floor."""
        rng = np.random.default_rng(3)
        floor = np.column_stack([rng.uniform(0, 4, 3000), np.zeros(3000), rng.uniform(0, 3, 3000)])
        # A deliberately denser wall, so "largest plane" alone would choose it.
        wall = np.column_stack([rng.uniform(0, 4, 12000), rng.uniform(0, 2.6, 12000), np.zeros(12000)])
        points = np.vstack([floor, wall]) + rng.normal(0, 0.003, (15000, 3))

        plane, _ = alignment.fit_plane_ransac(points, max_iterations=300, max_tilt_deg=35.0)
        assert abs(plane[1]) > 0.9, "the chosen plane should be near-horizontal"

    def test_refit_is_available_and_consistent(self):
        points = _box_room(seed=7)
        plane, inliers = alignment.fit_plane_ransac(points, max_iterations=200, max_tilt_deg=35.0, refine=True)
        assert abs(plane[1]) > 0.98
        assert len(inliers) > 500

    def test_rejects_too_few_points(self):
        with pytest.raises(alignment.AlignmentError, match="At least 3 points"):
            alignment.fit_plane_ransac(np.zeros((2, 3)))

    def test_rejects_wrong_shape(self):
        with pytest.raises(alignment.AlignmentError, match="Nx3"):
            alignment.fit_plane_ransac(np.zeros((10, 2)))

    def test_degenerate_cloud_raises_instead_of_returning_none(self):
        """
        A cloud of identical points used to return np.array(None) and blow up
        later with an unreadable error; it must fail here, clearly.
        """
        points = np.zeros((100, 3))
        with pytest.raises(alignment.AlignmentError):
            alignment.fit_plane_ransac(points, max_iterations=20)

    def test_outlier_filter_handles_tiny_clouds(self):
        """k must be clamped to the cloud size or every point looks like an outlier."""
        points = np.random.default_rng(1).random((5, 3))
        colors = np.zeros((5, 3), dtype=np.uint8)
        kept_points, kept_colors = alignment.remove_statistical_outliers(
            points, colors, nb_neighbors=20
        )
        assert len(kept_points) >= 1
        assert len(kept_points) == len(kept_colors)

    def test_outlier_filter_rejects_empty_cloud(self):
        with pytest.raises(alignment.AlignmentError, match="empty"):
            alignment.remove_statistical_outliers(np.empty((0, 3)), np.empty((0, 3)))


class TestFloorExtraction:
    def test_measures_a_known_room(self):
        points = _box_room(length=6.0, width=4.5)
        slice_points = vectorize.extract_floor_slice(points)
        polygon, extents = vectorize.extract_boundary_walls_polygon(slice_points)

        xs = [p["x"] for p in polygon]
        zs = [p["z"] for p in polygon]
        assert max(xs) - min(xs) == pytest.approx(6.0, abs=0.12)
        assert max(zs) - min(zs) == pytest.approx(4.5, abs=0.12)
        assert len(extents) >= 4

    def test_rejects_a_sparse_wall_band(self):
        with pytest.raises(vectorize.FloorExtractionError, match="too sparse"):
            vectorize.extract_boundary_walls_polygon(np.zeros((10, 3)))

    def test_rejects_non_finite_coordinates(self):
        points = _box_room(n=3000)
        band = vectorize.extract_floor_slice(points)
        band[0, 0] = np.nan
        band[:, 0] = np.nan
        with pytest.raises(vectorize.FloorExtractionError, match="NaN"):
            vectorize.extract_boundary_walls_polygon(band)

    def test_rejects_wrong_shape(self):
        with pytest.raises(vectorize.FloorExtractionError, match="Nx3"):
            vectorize.extract_floor_slice(np.zeros((10, 2)))

    def test_furniture_needs_a_real_polygon(self):
        with pytest.raises(vectorize.FloorExtractionError, match="at least 3 corners"):
            vectorize.detect_furniture_footprints(
                np.zeros((100, 3)), [{"x": 0, "z": 0}, {"x": 1, "z": 1}]
            )

    def test_grid_is_bounded_for_an_absurd_cloud(self):
        """
        A cloud in millimetres instead of metres used to request a grid with
        billions of cells. It must coarsen instead of exhausting memory.
        """
        w, h, res = vectorize._grid_shape(50_000.0, 50_000.0, 0.05, 0.4)
        assert w * h <= vectorize.MAX_GRID_CELLS
        assert res > 0.05, "resolution should have been coarsened"

    def test_detects_furniture_inside_the_room(self):
        rng = np.random.default_rng(5)
        room = _box_room(n=9000, seed=5)
        table = np.column_stack([
            rng.uniform(2.0, 3.2, 1800), rng.uniform(0.3, 0.75, 1800), rng.uniform(1.8, 2.8, 1800),
        ])
        points = np.vstack([room, table])
        polygon, _ = vectorize.extract_boundary_walls_polygon(
            vectorize.extract_floor_slice(points)
        )
        furniture = vectorize.detect_furniture_footprints(points, polygon, y_floor=0.0)
        assert len(furniture) >= 1
        assert any(f["bbox"]["width_m"] > 0.5 for f in furniture)
