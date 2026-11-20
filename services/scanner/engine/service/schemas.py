"""
Response models.

These are the API's contract with the frontend: documented in /docs, validated
on the way out, and the reason a field rename shows up as a test failure rather
than as an empty panel in the viewer.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Progress(BaseModel):
    step: int = Field(description="Completed pipeline stages, 0-6")
    total: int = Field(description="Total pipeline stages")
    message: str = Field(description="Human-readable current stage")


class ScanSummary(BaseModel):
    """Metrics extracted from a finished floorplan, for list rows and badges."""

    area_sqm: float | None = None
    area_sqft: float | None = None
    perimeter_m: float | None = None
    ceiling_height_m: float | None = None
    wall_count: int | None = None
    door_count: int | None = None
    window_count: int | None = None
    furniture_count: int | None = None
    camera_view_count: int | None = None
    point_count_fused: int | None = None


class Scan(BaseModel):
    id: str
    name: str
    source: str = Field(description="How the scan was created: 'upload' or 'sample'")
    status: str = Field(description="queued | processing | ready | failed")
    created_at: str
    updated_at: str
    progress: Progress
    summary: ScanSummary | None = None
    error: str | None = None
    duration_s: float | None = None
    artifacts: list[str] = Field(default_factory=list, description="Artifact names available for download")


class ScanList(BaseModel):
    scans: list[Scan]
    total: int


class Health(BaseModel):
    status: str
    service: str
    version: str
    worker: dict


class Config(BaseModel):
    auth_required: bool
    max_upload_mb: int
    max_scans: int
    sample_available: bool
    version: str


class Deleted(BaseModel):
    deleted: str


class ErrorBody(BaseModel):
    """Every non-2xx response has this shape, so the client has one error path."""

    error: str = Field(description="Short machine-readable code")
    detail: str = Field(description="Human-readable explanation, safe to display")
