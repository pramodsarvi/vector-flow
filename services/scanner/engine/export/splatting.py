"""
3D Gaussian Splatting (3DGS) Preparation & Training Wrapper
Converts Camera Poses and LiDAR Depth Point Clouds into standard
NeRF / 3DGS transforms.json dataset and prepares 3D Gaussian initialization.
"""

import json
import logging
import math
import os

log = logging.getLogger(__name__)

# Used when a capture bundle carries poses but no camera intrinsics. A 60-degree
# horizontal field of view on a 1920x1440 frame is a reasonable stand-in for a
# phone-class rig; the resulting transforms.json is still a usable dataset, it
# is just not calibrated.
FALLBACK_INTRINSICS = {"width": 1920, "height": 1440, "fx": 1662.77, "fy": 1662.77}

def _resolve_intrinsics(first_frame: dict) -> dict:
    """
    Camera intrinsics from the capture bundle, falling back to a sensible
    default. A missing calibration should degrade the splatting dataset, not
    abort a reconstruction that has already completed every other stage.
    """
    intr = dict(first_frame.get("intrinsics") or {})
    missing = [key for key in ("width", "height", "fx", "fy") if key not in intr]
    if missing:
        log.warning(
            "camera_poses.json is missing intrinsics (%s); using a default %dx%d pinhole model. "
            "The 3DGS dataset will not be metrically calibrated.",
            ", ".join(missing), FALLBACK_INTRINSICS["width"], FALLBACK_INTRINSICS["height"],
        )
        intr = {**FALLBACK_INTRINSICS, **intr}
    # The principal point is optional in most capture formats; the image centre
    # is the correct assumption when it is absent.
    intr.setdefault("cx", intr["width"] / 2.0)
    intr.setdefault("cy", intr["height"] / 2.0)
    return intr


def prepare_3dgs_dataset(scan_dir: str, output_dir: str):
    """
    Transforms camera poses and LiDAR cloud into NeRF/3DGS format.
    Generates transforms.json with 4x4 camera-to-world matrices.
    """
    os.makedirs(output_dir, exist_ok=True)
    poses_path = os.path.join(scan_dir, "camera_poses.json")

    if not os.path.exists(poses_path):
        raise FileNotFoundError(f"Poses file not found: {poses_path}")

    with open(poses_path) as f:
        meta = json.load(f)

    frames = meta.get("frames", [])
    if not frames:
        raise ValueError("No frames found in camera poses manifest")

    intr = _resolve_intrinsics(frames[0])

    # Compute field of view
    w, h = intr["width"], intr["height"]
    fx, fy = intr["fx"], intr["fy"]
    if not (w > 0 and h > 0 and fx > 0 and fy > 0):
        raise ValueError("Camera intrinsics must have positive width, height, fx and fy.")
    camera_angle_x = 2.0 * math.atan(w / (2.0 * fx))
    camera_angle_y = 2.0 * math.atan(h / (2.0 * fy))

    out_frames = []
    for f in frames:
        pos = f["camera_position"]
        orient = f.get("camera_orientation_yaw_pitch_roll", [0, 0, 0])
        yaw = math.radians(orient[0])
        pitch = math.radians(orient[1])
        roll = math.radians(orient[2])

        # Rotation matrix (Euler ZYX)
        cy, sy = math.cos(yaw), math.sin(yaw)
        cp, sp = math.cos(pitch), math.sin(pitch)
        cr, sr = math.cos(roll), math.sin(roll)

        R = [
            [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
            [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
            [-sp,   cp*sr,            cp*cr]
        ]

        # 4x4 transformation matrix [R | t]
        c2w = [
            [R[0][0], R[0][1], R[0][2], pos[0]],
            [R[1][0], R[1][1], R[1][2], pos[1]],
            [R[2][0], R[2][1], R[2][2], pos[2]],
            [0.0,     0.0,     0.0,     1.0]
        ]

        out_frames.append({
            "file_path": f["image_file"],
            "transform_matrix": c2w
        })

    transforms_data = {
        "camera_angle_x": camera_angle_x,
        "camera_angle_y": camera_angle_y,
        "fl_x": fx,
        "fl_y": fy,
        "cx": intr["cx"],
        "cy": intr["cy"],
        "w": w,
        "h": h,
        "frames": out_frames
    }

    transforms_path = os.path.join(output_dir, "transforms.json")
    with open(transforms_path, "w") as f:
        json.dump(transforms_data, f, indent=2)

    log.info("generated 3DGS transforms.json (%d camera views) at %s", len(out_frames), transforms_path)
    return transforms_path

if __name__ == "__main__":
    import argparse

    _ENGINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser(description="Build a NeRF/3DGS transforms.json from a scan bundle.")
    parser.add_argument("scan_dir", nargs="?",
                        default=os.path.join(os.path.dirname(_ENGINE), "capture", "sample-data"))
    parser.add_argument("output_dir", nargs="?", default=os.path.join(_ENGINE, "output", "3dgs"))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    print(prepare_3dgs_dataset(args.scan_dir, args.output_dir))
