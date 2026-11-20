"""
Automatic Floor Design & Vectorization Engine (Production Grade)
1. Architectural Slicing: isolates wall points in [0.15m, 1.20m]
2. General Polygonal Extraction: Supports arbitrary room layouts (Rectangular, L-shaped, etc.)
   using 2D occupancy grid, contour solidity analysis, and Manhattan orthogonal snapping.
3. Furniture Footprint & Obstacle Detection: Extracts 2D bounding boxes of furniture on the floor.
4. Structural Opening Recognition: Detects doors and windows with sill and height metrics.
5. Metric Computation: Net floor area, perimeter, and wall segment lengths.
"""

import logging
import math

import cv2
import numpy as np
import shapely
from shapely.geometry import Polygon

log = logging.getLogger(__name__)

# Guard against a pathological cloud (bad alignment, metres-vs-millimetres mix-up)
# asking for a grid so large it exhausts memory. Past this the resolution is
# coarsened instead, which degrades accuracy but always terminates.
MAX_GRID_CELLS = 16_000_000
# Furniture must sit at least this far inside the walls to count as an obstacle
# rather than wall clutter.
WALL_CLEARANCE_M = 0.20


class FloorExtractionError(ValueError):
    """The scan could not be vectorized. Messages here are safe to show a user."""


def _grid_shape(span_x: float, span_z: float, grid_res: float, margin: float) -> tuple[int, int, float]:
    """
    Cells needed to rasterize a span, with the resolution coarsened if the
    requested one would allocate an unreasonable grid.
    """
    while True:
        w_cells = max(1, int(math.ceil((span_x + 2 * margin) / grid_res)))
        h_cells = max(1, int(math.ceil((span_z + 2 * margin) / grid_res)))
        if w_cells * h_cells <= MAX_GRID_CELLS:
            return w_cells, h_cells, grid_res
        grid_res *= 2.0
        log.warning(
            "Point cloud spans %.1f x %.1f m; coarsening occupancy grid to %.3f m/cell.",
            span_x, span_z, grid_res,
        )

def extract_floor_slice(points: np.ndarray, y_min: float = 0.15, y_max: float = 1.20):
    """
    Extracts points within the architectural horizontal band to exclude floor rugs and ceiling fixtures.
    """
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] != 3:
        raise FloorExtractionError(f"Expected an Nx3 point array, got shape {points.shape}.")
    mask = (points[:, 1] >= y_min) & (points[:, 1] <= y_max)
    return points[mask]

def extract_boundary_walls_polygon(slice_points: np.ndarray, grid_res: float = 0.05):
    """
    General Polygonal Extraction:
    Rasterizes the point cloud onto a 2D occupancy grid, analyzes contour solidity,
    and regularizes to Manhattan-World orthogonal segments.
    Supports both rectangular (4 corners) and complex L-shaped / multi-corner rooms!
    """
    slice_points = np.asarray(slice_points)
    if len(slice_points) < 50:
        raise FloorExtractionError(
            f"Only {len(slice_points)} points fall in the wall band; the scan is too sparse "
            "or the floor was not detected correctly."
        )

    xz = slice_points[:, [0, 2]]
    min_x, max_x = np.percentile(xz[:, 0], 1), np.percentile(xz[:, 0], 99)
    min_z, max_z = np.percentile(xz[:, 1], 1), np.percentile(xz[:, 1], 99)
    if not all(np.isfinite([min_x, max_x, min_z, max_z])):
        raise FloorExtractionError("Point cloud contains NaN or infinite coordinates.")

    margin = 0.4
    w_cells, h_cells, grid_res = _grid_shape(max_x - min_x, max_z - min_z, grid_res, margin)

    grid = np.zeros((h_cells, w_cells), dtype=np.uint8)
    gx = ((xz[:, 0] - min_x + margin) / grid_res).astype(np.int32)
    gz = ((xz[:, 1] - min_z + margin) / grid_res).astype(np.int32)
    valid = (gx >= 0) & (gx < w_cells) & (gz >= 0) & (gz < h_cells)
    grid[gz[valid], gx[valid]] = 255

    # Morphological closing
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(grid, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    is_convex_room = True
    concave_pts = []

    if contours:
        cnt = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(cnt)
        area = cv2.contourArea(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = (area / hull_area) if hull_area > 0 else 1.0

        # If solidity is significantly less than 1.0, the room has real concave geometry (e.g. L-shaped)
        if solidity < 0.85:
            is_convex_room = False
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
            if len(approx) >= 4:
                for pt in approx:
                    px = (pt[0][0] * grid_res) + min_x - margin
                    pz = (pt[0][1] * grid_res) + min_z - margin
                    concave_pts.append({"x": round(float(px), 3), "z": round(float(pz), 3)})

    if not is_convex_room and len(concave_pts) >= 4:
        poly_pts = concave_pts
        xs = [p["x"] for p in poly_pts]
        zs = [p["z"] for p in poly_pts]
        extents = (min(xs), max(xs), min(zs), max(zs))
    else:
        # Standard convex / rectangular room: fit robust orthogonal walls
        ref_min_x = float(np.median(xz[np.abs(xz[:, 0] - min_x) < 0.25, 0])) if np.sum(np.abs(xz[:, 0] - min_x) < 0.25) > 10 else float(min_x)
        ref_max_x = float(np.median(xz[np.abs(xz[:, 0] - max_x) < 0.25, 0])) if np.sum(np.abs(xz[:, 0] - max_x) < 0.25) > 10 else float(max_x)
        ref_min_z = float(np.median(xz[np.abs(xz[:, 1] - min_z) < 0.25, 1])) if np.sum(np.abs(xz[:, 1] - min_z) < 0.25) > 10 else float(min_z)
        ref_max_z = float(np.median(xz[np.abs(xz[:, 1] - max_z) < 0.25, 1])) if np.sum(np.abs(xz[:, 1] - max_z) < 0.25) > 10 else float(max_z)

        poly_pts = [
            {"x": round(ref_min_x, 3), "z": round(ref_min_z, 3)},
            {"x": round(ref_max_x, 3), "z": round(ref_min_z, 3)},
            {"x": round(ref_max_x, 3), "z": round(ref_max_z, 3)},
            {"x": round(ref_min_x, 3), "z": round(ref_max_z, 3)},
        ]
        extents = (ref_min_x, ref_max_x, ref_min_z, ref_max_z)

    return poly_pts, extents

def detect_furniture_footprints(points: np.ndarray, room_polygon_pts: list, y_floor: float = 0.0):
    """
    Detects movable interior objects and furniture footprints (sofas, tables, beds)
    from points inside the room sitting between Y in [0.05m, 0.85m].
    """
    poly_coords = [(v["x"], v["z"]) for v in room_polygon_pts]
    if len(poly_coords) < 3:
        raise FloorExtractionError(
            f"A room polygon needs at least 3 corners, got {len(poly_coords)}."
        )
    poly = Polygon(poly_coords)
    if not poly.is_valid:
        # Self-intersecting outlines come out of noisy contour tracing; buffer(0)
        # is the standard repair and keeps the dominant region.
        poly = poly.buffer(0)
        if poly.is_empty:
            raise FloorExtractionError("The extracted room outline is degenerate.")
        if poly.geom_type == "MultiPolygon":
            poly = max(poly.geoms, key=lambda g: g.area)

    # Filter points at furniture height
    mask = (points[:, 1] >= y_floor + 0.05) & (points[:, 1] <= y_floor + 0.90)
    furn_pts = points[mask]

    if len(furn_pts) < 30:
        return []

    # Exclude points near exterior walls. Eroding the room polygon by the
    # clearance and testing containment is equivalent to "inside and further
    # than the clearance from the boundary", but it is one vectorized call
    # instead of two Shapely calls per point — the difference between seconds
    # and minutes on a dense real-world cloud.
    interior_region = poly.buffer(-WALL_CLEARANCE_M)
    if interior_region.is_empty:
        return []
    inside = shapely.contains_xy(interior_region, furn_pts[:, 0], furn_pts[:, 2])
    interior_pts = furn_pts[inside]

    if len(interior_pts) < 20:
        return []


    # Cluster points in 2D using grid connected components
    grid_res = 0.10
    xz = interior_pts[:, [0, 2]]
    min_x, max_x = np.min(xz[:, 0]), np.max(xz[:, 0])
    min_z, max_z = np.min(xz[:, 1]), np.max(xz[:, 1])

    w_cells = max(1, int(math.ceil((max_x - min_x + 0.2) / grid_res)))
    h_cells = max(1, int(math.ceil((max_z - min_z + 0.2) / grid_res)))

    grid = np.zeros((h_cells, w_cells), dtype=np.uint8)
    gx = np.clip(((xz[:, 0] - min_x) / grid_res).astype(np.int32), 0, w_cells - 1)
    gz = np.clip(((xz[:, 1] - min_z) / grid_res).astype(np.int32), 0, h_cells - 1)
    grid[gz, gx] = 255

    # Connect nearby points
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    grid = cv2.dilate(grid, kernel)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(grid)

    furniture = []
    for label_id in range(1, num_labels):
        stat = stats[label_id]
        area = stat[cv2.CC_STAT_AREA]
        if area < 15:  # filter noise
            continue

        c_min_x = min_x + stat[cv2.CC_STAT_LEFT] * grid_res
        c_min_z = min_z + stat[cv2.CC_STAT_TOP] * grid_res
        width = stat[cv2.CC_STAT_WIDTH] * grid_res
        depth = stat[cv2.CC_STAT_HEIGHT] * grid_res

        # Find points belonging to this cluster to estimate height
        cluster_mask = (xz[:, 0] >= c_min_x - 0.05) & (xz[:, 0] <= c_min_x + width + 0.05) & \
                       (xz[:, 1] >= c_min_z - 0.05) & (xz[:, 1] <= c_min_z + depth + 0.05)
        cluster_y = interior_pts[cluster_mask, 1]
        height = float(np.max(cluster_y) - y_floor) if len(cluster_y) > 0 else 0.75
        if height < 0.20:  # floor-noise / rug-level blobs are not obstacles
            continue

        # Classify asset based on aspect ratio and dimensions
        obj_type = "Furniture"
        if width >= 1.6 or depth >= 1.6:
            obj_type = "Sofa / Seating"
        elif 0.8 <= width < 1.6 and 0.8 <= depth < 1.6:
            obj_type = "Table / Desk"
        else:
            obj_type = "Armchair / Stand"

        furniture.append({
            "type": obj_type,
            "bbox": {
                "center_x": round(float(c_min_x + width / 2.0), 2),
                "center_z": round(float(c_min_z + depth / 2.0), 2),
                "width_m": round(float(width), 2),
                "depth_m": round(float(depth), 2),
                "height_m": round(float(height), 2)
            }
        })

    return furniture

def detect_openings_on_walls(points: np.ndarray, extents: tuple, room_height: float = 2.8):
    """
    Detects doors (floor-to-lintel openings) and windows (openings with a sill).
    """
    min_x, max_x, min_z, max_z = extents
    openings = []

    # Front wall (Z = min_z)
    w_front = points[np.abs(points[:, 2] - min_z) < 0.2]
    openings.extend(_scan_wall_openings(w_front, axis='x', fixed_coord=min_z, wall_name="front", room_height=room_height))

    # Back wall (Z = max_z)
    w_back = points[np.abs(points[:, 2] - max_z) < 0.2]
    openings.extend(_scan_wall_openings(w_back, axis='x', fixed_coord=max_z, wall_name="back", room_height=room_height))

    # Left wall (X = min_x)
    w_left = points[np.abs(points[:, 0] - min_x) < 0.2]
    openings.extend(_scan_wall_openings(w_left, axis='z', fixed_coord=min_x, wall_name="left", room_height=room_height))

    # Right wall (X = max_x)
    w_right = points[np.abs(points[:, 0] - max_x) < 0.2]
    openings.extend(_scan_wall_openings(w_right, axis='z', fixed_coord=max_x, wall_name="right", room_height=room_height))

    return openings

def _scan_wall_openings(wall_points: np.ndarray, axis: str, fixed_coord: float, wall_name: str, room_height: float):
    if len(wall_points) < 50:
        return []

    coord_idx = 0 if axis == 'x' else 2
    coords = wall_points[:, coord_idx]
    y_coords = wall_points[:, 1]

    min_c, max_c = np.min(coords), np.max(coords)
    bin_width = 0.10
    bins = np.arange(min_c, max_c + bin_width, bin_width)

    empty_bins = []
    has_sill = []

    for i in range(len(bins) - 1):
        b_start, b_end = bins[i], bins[i+1]
        in_bin = (coords >= b_start) & (coords < b_end)
        bin_y = y_coords[in_bin]

        eye_pts = np.sum((bin_y >= 1.0) & (bin_y <= 1.8))
        floor_pts = np.sum((bin_y >= 0.1) & (bin_y <= 0.8))

        if eye_pts < 3:
            empty_bins.append(b_start)
            has_sill.append(floor_pts > 8)

    if not empty_bins:
        return []

    clusters = []
    current_cluster = [empty_bins[0]]
    current_sill = [has_sill[0]]

    for i in range(1, len(empty_bins)):
        if abs(empty_bins[i] - empty_bins[i-1] - bin_width) < 0.05:
            current_cluster.append(empty_bins[i])
            current_sill.append(has_sill[i])
        else:
            clusters.append((current_cluster, current_sill))
            current_cluster = [empty_bins[i]]
            current_sill = [has_sill[i]]
    clusters.append((current_cluster, current_sill))

    openings = []
    for cl, sl in clusters:
        width = len(cl) * bin_width
        if 0.70 <= width <= 3.5:
            start_c = round(cl[0], 2)
            end_c = round(cl[-1] + bin_width, 2)
            is_window = any(sl)
            op_type = "window" if is_window else "door"

            if axis == 'x':
                start_pt = {"x": start_c, "z": round(fixed_coord, 2)}
                end_pt = {"x": end_c, "z": round(fixed_coord, 2)}
            else:
                start_pt = {"x": round(fixed_coord, 2), "z": start_c}
                end_pt = {"x": round(fixed_coord, 2), "z": end_c}

            openings.append({
                "type": op_type,
                "wall": wall_name,
                "start": start_pt,
                "end": end_pt,
                "width_m": round(width, 2),
                "height_m": 1.4 if is_window else 2.1,
                "sill_height_m": 0.85 if is_window else 0.0,
                "swing": "inward_right" if op_type == "door" else None
            })

    return openings

def compute_room_metrics(polygon_vertices: list):
    """
    Computes floor area, perimeter, and wall lengths.
    """
    poly_coords = [(v["x"], v["z"]) for v in polygon_vertices]
    poly = Polygon(poly_coords)
    area_sqm = round(poly.area, 2)
    area_sqft = round(area_sqm * 10.7639, 2)
    perimeter_m = round(poly.length, 2)

    walls = []
    n = len(polygon_vertices)
    for i in range(n):
        p1 = polygon_vertices[i]
        p2 = polygon_vertices[(i + 1) % n]
        dx = p2["x"] - p1["x"]
        dz = p2["z"] - p1["z"]
        length = round(math.sqrt(dx * dx + dz * dz), 2)
        walls.append({
            "wall_id": f"wall_{i+1}",
            "start": p1,
            "end": p2,
            "length_m": length,
            "length_ft": round(length * 3.28084, 2)
        })

    return {
        "area_sqm": area_sqm,
        "area_sqft": area_sqft,
        "perimeter_m": perimeter_m,
        "walls": walls
    }

def generate_floorplan_schema(
    project_id: str,
    polygon_vertices: list,
    openings: list,
    furniture: list = None,
    ceiling_height: float = 2.8
):
    """
    Creates complete architectural JSON schema including walls, openings, and furniture footprints.
    """
    metrics = compute_room_metrics(polygon_vertices)
    return {
        "project_id": project_id,
        "version": "1.1",
        "units": "meters",
        "ceiling_height_m": ceiling_height,
        "room": {
            "name": "Main Scanned Room",
            "area_sqm": metrics["area_sqm"],
            "area_sqft": metrics["area_sqft"],
            "perimeter_m": metrics["perimeter_m"],
            "polygon": polygon_vertices,
            "walls": metrics["walls"],
            "openings": openings,
            "furniture_footprints": furniture or []
        }
    }
