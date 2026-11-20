"""
Runtime configuration for the Spatial Scanner service.

Every knob is an environment variable so the same image runs on a laptop and
in a container without code changes. Read once at import; `settings` is the
single instance the rest of the service uses.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ENGINE_DIR = Path(__file__).resolve().parent.parent
PRODUCT_DIR = ENGINE_DIR.parent


def _env_path(name: str, default: Path) -> Path:
    raw = os.environ.get(name, "").strip()
    return Path(raw).expanduser().resolve() if raw else default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_list(name: str, default: list[str]) -> list[str]:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    return [item.strip() for item in raw.split(",") if item.strip()]


@dataclass(frozen=True)
class Settings:
    """Service configuration, resolved from the environment."""

    # Where scans live. One directory per scan: data/scans/<scan_id>/{input,output}
    data_dir: Path = field(default_factory=lambda: _env_path("SCANNER_DATA", PRODUCT_DIR / "data"))
    # Bundled demo scan used by POST /api/scans/sample
    sample_dir: Path = field(
        default_factory=lambda: _env_path("SCANNER_SAMPLE", PRODUCT_DIR / "capture" / "sample-data")
    )
    # Static frontend
    viewer_dir: Path = field(default_factory=lambda: _env_path("SCANNER_VIEWER", PRODUCT_DIR / "viewer"))

    # Shared-secret auth. Empty = open (fine on localhost, not beyond it).
    token: str = field(default_factory=lambda: os.environ.get("SCANNER_TOKEN", "").strip())
    # Browser origins allowed to call the API. Default same-origin only.
    cors_origins: list[str] = field(default_factory=lambda: _env_list("SCANNER_CORS_ORIGINS", []))

    # Upload ceiling, megabytes. Point clouds are large but not unbounded.
    max_upload_mb: int = field(default_factory=lambda: _env_int("SCANNER_MAX_UPLOAD_MB", 512))
    # Scans kept on disk; the oldest finished ones are pruned past this.
    max_scans: int = field(default_factory=lambda: _env_int("SCANNER_MAX_SCANS", 50))
    log_level: str = field(default_factory=lambda: os.environ.get("SCANNER_LOG", "INFO").upper())

    @property
    def scans_dir(self) -> Path:
        return self.data_dir / "scans"

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    def public(self) -> dict:
        """Config the frontend is allowed to know about."""
        return {
            "auth_required": bool(self.token),
            "max_upload_mb": self.max_upload_mb,
            "max_scans": self.max_scans,
            "sample_available": (self.sample_dir / "raw_lidar_cloud.ply").exists(),
        }


settings = Settings()
