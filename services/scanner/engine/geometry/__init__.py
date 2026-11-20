"""
Turning a raw point cloud into room geometry.

`alignment` finds the floor and levels the capture onto it; `vectorize` slices
the levelled cloud into walls, openings and furniture footprints, and builds
the floorplan schema everything downstream consumes.
"""

from .alignment import AlignmentError, align_floor_to_y_up, remove_statistical_outliers
from .vectorize import (
    FloorExtractionError,
    detect_furniture_footprints,
    detect_openings_on_walls,
    extract_boundary_walls_polygon,
    extract_floor_slice,
    generate_floorplan_schema,
)

__all__ = [
    "AlignmentError", "align_floor_to_y_up", "remove_statistical_outliers",
    "FloorExtractionError", "extract_floor_slice", "extract_boundary_walls_polygon",
    "detect_openings_on_walls", "detect_furniture_footprints", "generate_floorplan_schema",
]
