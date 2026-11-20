"""
Turning an HTTP upload into a scan bundle on disk.

Two rules this module exists to enforce:
  * nothing is written past the configured size cap, checked while streaming
    rather than after the fact, so a huge upload cannot fill the disk;
  * a point cloud is validated by parsing its header, not by trusting its
    filename, so `malware.ply` is rejected on content.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from pathlib import Path
from typing import BinaryIO

from ..ingest.ply_io import PlyError, validate_file
from .settings import settings

log = logging.getLogger(__name__)

CHUNK = 1024 * 1024
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
MAX_FRAMES = 2000


class IngestError(ValueError):
    """The upload was rejected. Messages here are safe to show a user."""


def stream_to_file(src: BinaryIO, dest: Path, max_bytes: int) -> int:
    """
    Copy an upload to disk, aborting the moment it exceeds max_bytes.

    Returns the number of bytes written. On overflow the partial file is
    removed and IngestError raised.
    """
    written = 0
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(dest, "wb") as out:
            while True:
                chunk = src.read(CHUNK)
                if not chunk:
                    break
                written += len(chunk)
                if written > max_bytes:
                    raise IngestError(
                        f"Upload exceeds the {max_bytes // (1024 * 1024)} MB limit."
                    )
                out.write(chunk)
    except IngestError:
        dest.unlink(missing_ok=True)
        raise
    except OSError as exc:
        dest.unlink(missing_ok=True)
        raise IngestError(f"Could not store the upload: {exc.strerror or exc}") from exc
    if written == 0:
        dest.unlink(missing_ok=True)
        raise IngestError("Upload was empty.")
    return written


def accept_point_cloud(src: BinaryIO, input_dir: Path) -> dict:
    """
    Store an uploaded point cloud as raw_lidar_cloud.ply and validate it.

    Validation happens after writing because a PLY header can legitimately sit
    beyond the first chunk, and because validating the file on disk is what the
    pipeline will actually read.
    """
    dest = input_dir / "raw_lidar_cloud.ply"
    size = stream_to_file(src, dest, settings.max_upload_bytes)
    try:
        info = validate_file(str(dest))
    except PlyError as exc:
        dest.unlink(missing_ok=True)
        raise IngestError(str(exc)) from exc
    log.info(
        "accepted point cloud: %s format, %d vertices, %.1f MB",
        info["format"], info["vertex_count"], size / 1e6,
    )
    return {"bytes": size, **info}


def accept_poses(src: BinaryIO, input_dir: Path) -> int:
    """Store an optional camera_poses.json, rejecting anything that is not valid JSON."""
    dest = input_dir / "camera_poses.json"
    # Poses are small; a few MB is already generous for thousands of frames.
    stream_to_file(src, dest, min(settings.max_upload_bytes, 32 * 1024 * 1024))
    try:
        data = json.loads(dest.read_text())
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        dest.unlink(missing_ok=True)
        raise IngestError("camera_poses.json is not valid JSON.") from exc
    frames = data.get("frames") if isinstance(data, dict) else None
    if not isinstance(frames, list):
        dest.unlink(missing_ok=True)
        raise IngestError("camera_poses.json must be an object with a 'frames' array.")
    if len(frames) > MAX_FRAMES:
        dest.unlink(missing_ok=True)
        raise IngestError(f"camera_poses.json declares {len(frames)} frames; the limit is {MAX_FRAMES}.")
    return len(frames)


def accept_frame(src: BinaryIO, input_dir: Path, filename: str) -> str:
    """
    Store one capture frame under input/images/.

    The client-supplied filename is reduced to a basename with an allowlisted
    extension, so it can never escape the scan directory.
    """
    base = os.path.basename(filename or "")
    suffix = Path(base).suffix.lower()
    if not base or base.startswith(".") or suffix not in IMAGE_SUFFIXES:
        raise IngestError(f"Unsupported frame image '{filename}'. Use .jpg or .png.")
    images = input_dir / "images"
    stream_to_file(src, images / base, settings.max_upload_bytes)
    return base


def copy_sample_bundle(input_dir: Path) -> dict:
    """Populate a scan from the bundled demo capture, for a one-click first run."""
    sample = settings.sample_dir
    cloud = sample / "raw_lidar_cloud.ply"
    if not cloud.is_file():
        raise IngestError(f"No sample scan is installed at {sample}.")

    input_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(cloud, input_dir / "raw_lidar_cloud.ply")

    poses = sample / "camera_poses.json"
    if poses.is_file():
        shutil.copy2(poses, input_dir / "camera_poses.json")

    frames = 0
    images_src = sample / "images"
    if images_src.is_dir():
        dest = input_dir / "images"
        dest.mkdir(exist_ok=True)
        for image in sorted(images_src.iterdir()):
            if image.suffix.lower() in IMAGE_SUFFIXES:
                shutil.copy2(image, dest / image.name)
                frames += 1
    return {"frames": frames}
