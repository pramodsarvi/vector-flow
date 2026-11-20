"""
Spatial Scanner processing pipeline.

Orchestrates the engine's building blocks in order:

    ingest    -> open and validate the capture bundle, read the point cloud
    geometry  -> level the cloud onto its floor, vectorize walls and contents
    export    -> write 2D CAD, 3D meshes, colliders and the splatting dataset

Usable three ways, all driving the same code path:
  * as a library   -- `process_spatial_scan(input_dir, output_dir, progress=cb)`
  * from the API   -- the job worker passes a progress callback for status reporting
  * from the CLI   -- `python -m engine.pipeline [input_dir] [output_dir]`
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

from .export import (
    export_physics_colliders,
    export_textured_room_glb,
    export_to_dxf,
    export_to_pdf,
    export_to_svg,
    generate_architectural_3d_shell,
    prepare_3dgs_dataset,
)
from .geometry import (
    align_floor_to_y_up,
    detect_furniture_footprints,
    detect_openings_on_walls,
    extract_boundary_walls_polygon,
    extract_floor_slice,
    generate_floorplan_schema,
    remove_statistical_outliers,
)
from .ingest import ScanInputError, open_bundle, read_ply, write_ply

log = logging.getLogger(__name__)

TOTAL_STEPS = 6
DEFAULT_CEILING_HEIGHT_M = 2.8

ENGINE_DIR = Path(__file__).resolve().parent
PRODUCT_DIR = ENGINE_DIR.parent

# Repo-relative defaults, resolved from this file's location so the CLI works
# from any working directory.
DEFAULT_INPUT_DIR = str(PRODUCT_DIR / "capture" / "sample-data")
DEFAULT_OUTPUT_DIR = str(ENGINE_DIR / "output")

# A callback invoked as progress(step, total_steps, message).
ProgressFn = Callable[[int, int, str], None]


def process_spatial_scan(
    input_dir: str,
    output_dir: str,
    progress: ProgressFn | None = None,
    sync_viewer: bool = False,
    ceiling_height: float = DEFAULT_CEILING_HEIGHT_M,
    project_id: str | None = None,
) -> dict:
    """
    Executes the complete production pipeline on a captured scan bundle.

    progress: optional callback(step, total, message) for status reporting.
    sync_viewer: copy the 2D deliverables into viewer/ as well. Off by
        default — the API serves artifacts per scan, so only the standalone CLI
        needs this.

    project_id: identifier stamped into the schema. Defaults to the scan
        directory's name.

    Returns the floorplan schema dict, which is also written to floorplan.json.
    """
    start_time = time.time()
    report: ProgressFn = progress or (lambda step, total, message: None)

    os.makedirs(output_dir, exist_ok=True)
    np.random.seed(42)  # deterministic RANSAC -> reproducible measurements run-to-run

    bundle = open_bundle(input_dir)
    log.info("ingesting scan from %s", input_dir)

    # 1. Load Point Cloud & Poses
    report(1, TOTAL_STEPS, "Loading point cloud & sensor data")
    points, colors = read_ply(str(bundle.cloud_path))
    log.info("loaded %d points with RGB color attributes", len(points))

    # 2. Alignment & Coordinate Normalization
    report(2, TOTAL_STEPS, "Detecting floor plane & normalizing coordinates")
    aligned_points, T_matrix, floor_eq = align_floor_to_y_up(points)
    cleaned_points, cleaned_colors = remove_statistical_outliers(aligned_points, colors)

    aligned_ply = os.path.join(output_dir, "aligned_point_cloud.ply")
    write_ply(aligned_ply, cleaned_points, cleaned_colors)
    log.info(
        "floor normal aligned to Y-up; removed %d outlier points",
        len(aligned_points) - len(cleaned_points),
    )

    # 3. Floor Plan Slicing & Manhattan-World Vectorization
    report(3, TOTAL_STEPS, "Slicing wall band & vectorizing walls")
    slice_points = extract_floor_slice(cleaned_points, y_min=0.15, y_max=1.20)
    polygon_vertices, extents = extract_boundary_walls_polygon(slice_points)
    log.info(
        "wall band holds %d points; extracted %d wall corners",
        len(slice_points), len(polygon_vertices),
    )

    # 4. Furniture Obstacle Detection
    report(4, TOTAL_STEPS, "Detecting floor clutter & furniture footprints")
    furniture = detect_furniture_footprints(cleaned_points, polygon_vertices, y_floor=0.0)
    log.info("detected %d furniture objects: %s", len(furniture), [f["type"] for f in furniture])

    # 5. Opening Detection (Doors & Windows)
    report(5, TOTAL_STEPS, "Detecting structural openings")
    openings = detect_openings_on_walls(cleaned_points, extents)
    log.info("detected %d openings: %s", len(openings), [o["type"] for o in openings])

    # Build schema
    floorplan = generate_floorplan_schema(
        project_id=project_id or _derive_project_id(output_dir),
        polygon_vertices=polygon_vertices,
        openings=openings,
        furniture=furniture,
        ceiling_height=ceiling_height,
    )
    floorplan["scan"] = {
        "point_count_raw": int(len(points)),
        "point_count_fused": int(len(cleaned_points)),
        "camera_views": bundle.camera_views(T_matrix),
    }

    # 6. Multi-Format CAD, PDF & 3D World Generation
    report(6, TOTAL_STEPS, "Exporting CAD, blueprint & 3D models")

    json_path = os.path.join(output_dir, "floorplan.json")
    with open(json_path, "w") as f:
        json.dump(floorplan, f, indent=2)

    export_to_svg(floorplan, os.path.join(output_dir, "floorplan.svg"))
    export_to_dxf(floorplan, os.path.join(output_dir, "floorplan.dxf"))
    export_to_pdf(floorplan, os.path.join(output_dir, "floorplan.pdf"))

    obj_path = os.path.join(output_dir, "room_model.obj")
    mesh = generate_architectural_3d_shell(floorplan, output_obj=obj_path)
    export_textured_room_glb(mesh, os.path.join(output_dir, "room_model.glb"))
    export_physics_colliders(floorplan, os.path.join(output_dir, "colliders.json"))

    # 3D Gaussian Splatting Dataset (transforms.json)
    if bundle.has_poses:
        prepare_3dgs_dataset(str(bundle.root), os.path.join(output_dir, "3dgs"))

    if sync_viewer:
        _sync_viewer(output_dir)

    elapsed = time.time() - start_time
    floorplan["processing_time_s"] = round(elapsed, 2)
    with open(json_path, "w") as f:
        json.dump(floorplan, f, indent=2)

    log.info(
        "pipeline completed in %.2fs — area %s m², %d furniture, perimeter %s m",
        elapsed, floorplan["room"]["area_sqm"], len(furniture), floorplan["room"]["perimeter_m"],
    )
    return floorplan


def _derive_project_id(output_dir: str) -> str:
    """
    Name the project after the scan rather than after its artifact folder:
    the store lays scans out as <scan_id>/output, so the basename alone would
    label every scan "output".
    """
    path = os.path.normpath(os.path.abspath(output_dir))
    name = os.path.basename(path)
    if name in ("output", "out", ""):
        name = os.path.basename(os.path.dirname(path))
    return name or "scan"


def _sync_viewer(output_dir: str) -> None:
    """Copy the 2D deliverables next to the standalone viewer (CLI convenience)."""
    viewer_dir = str(PRODUCT_DIR / "viewer")
    if not os.path.isdir(viewer_dir):
        return
    for name in ("floorplan.json", "floorplan.svg", "floorplan.pdf"):
        src = os.path.join(output_dir, name)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(viewer_dir, name))
    log.info("synced floorplan JSON/SVG/PDF into %s", viewer_dir)


def _cli_progress(step: int, total: int, message: str) -> None:
    print(f"Step {step}/{total}: {message}...")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Process a camera + LiDAR scan bundle into 2D CAD floor designs and a 3D room model."
    )
    parser.add_argument("input_dir", nargs="?", default=DEFAULT_INPUT_DIR,
                        help="scan bundle directory (raw_lidar_cloud.ply, camera_poses.json, images/)")
    parser.add_argument("output_dir", nargs="?", default=DEFAULT_OUTPUT_DIR,
                        help="where to write the generated artifacts")
    parser.add_argument("--sync-viewer", action="store_true",
                        help="also copy floorplan.json/svg/pdf into viewer/")
    parser.add_argument("--ceiling-height", type=float, default=DEFAULT_CEILING_HEIGHT_M,
                        help="assumed ceiling height in metres (default: %(default)s)")
    parser.add_argument("-q", "--quiet", action="store_true", help="only report warnings and errors")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    print("=" * 60)
    print(f"INGESTING SCAN: {args.input_dir}")
    print("=" * 60)
    try:
        floorplan = process_spatial_scan(
            args.input_dir,
            args.output_dir,
            progress=None if args.quiet else _cli_progress,
            sync_viewer=args.sync_viewer,
            ceiling_height=args.ceiling_height,
        )
    except (ScanInputError, ValueError) as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1

    room = floorplan["room"]
    print("=" * 60)
    print(f"PIPELINE COMPLETED IN {floorplan['processing_time_s']:.2f}s")
    print(f"   Room Area:      {room['area_sqm']} m² ({room['area_sqft']} sq ft)")
    print(f"   Perimeter:      {room['perimeter_m']} m")
    print(f"   Walls:          {len(room['walls'])}")
    print(f"   Openings:       {len(room['openings'])}")
    print(f"   Furniture:      {len(room['furniture_footprints'])}")
    print(f"   Artifacts:      {args.output_dir}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
