"""
Opening a capture bundle.

A bundle is a directory produced by a capture client:

    raw_lidar_cloud.ply     the point cloud (required)
    camera_poses.json       per-frame camera poses and intrinsics (optional)
    images/                 the captured RGB frames (optional)

Validation happens here, before the pipeline does any work, so a missing or
malformed bundle fails immediately with a message worth showing a user rather
than deep inside a numpy call.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

CLOUD_NAME = "raw_lidar_cloud.ply"
POSES_NAME = "camera_poses.json"
IMAGES_DIR = "images"


class ScanInputError(ValueError):
    """The bundle is missing or malformed. Messages here are safe to show a user."""


@dataclass(frozen=True)
class ScanBundle:
    """A validated capture bundle, with the paths the pipeline needs."""

    root: Path
    cloud_path: Path
    poses_path: Path
    images_dir: Path

    @property
    def has_poses(self) -> bool:
        return self.poses_path.is_file()

    def load_frames(self) -> list[dict]:
        """
        Camera frames from the poses file, or an empty list when there is none.

        A frame without a position is skipped rather than fatal: a partial pose
        log should cost you that viewpoint, not the whole reconstruction.
        """
        if not self.has_poses:
            return []
        try:
            data = json.loads(self.poses_path.read_text())
        except json.JSONDecodeError as exc:
            raise ScanInputError(f"{POSES_NAME} is not valid JSON: {exc.msg}") from exc
        except OSError as exc:
            raise ScanInputError(f"Could not read {POSES_NAME}: {exc.strerror or exc}") from exc

        if not isinstance(data, dict):
            raise ScanInputError(f"{POSES_NAME} must be a JSON object.")
        frames = data.get("frames", [])
        if not isinstance(frames, list):
            raise ScanInputError(f"{POSES_NAME} must contain a 'frames' array.")

        usable = [f for f in frames if isinstance(f, dict) and "camera_position" in f]
        if len(usable) < len(frames):
            log.warning("%d of %d frames have no camera_position and were skipped",
                        len(frames) - len(usable), len(frames))
        return usable

    def camera_views(self, transform: np.ndarray) -> list[dict]:
        """
        Camera viewpoints mapped into the aligned coordinate frame, so the
        viewer can place a pin where the photo was actually taken.
        """
        views = []
        for frame in self.load_frames():
            position = np.append(np.asarray(frame["camera_position"], dtype=float), 1.0)
            x, y, z = (transform @ position)[:3]
            views.append({
                "frame_id": frame.get("frame_id"),
                "image_file": frame.get("image_file"),
                "position": [round(float(x), 3), round(float(y), 3), round(float(z), 3)],
            })
        return views


def open_bundle(input_dir: str | Path) -> ScanBundle:
    """Validate a capture directory and return the paths within it."""
    root = Path(input_dir)
    if not root.is_dir():
        raise ScanInputError(f"Scan directory does not exist: {root}")

    cloud = root / CLOUD_NAME
    if not cloud.is_file():
        raise ScanInputError(f"No point cloud found at {CLOUD_NAME} in the scan directory.")

    return ScanBundle(
        root=root,
        cloud_path=cloud,
        poses_path=root / POSES_NAME,
        images_dir=root / IMAGES_DIR,
    )
