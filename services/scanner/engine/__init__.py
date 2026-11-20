"""
Spatial Scanner processing engine.

Four building blocks, each one stage of turning a capture into deliverables:

    ingest    reading a scan bundle off disk (point clouds, poses, frames)
    geometry  making sense of it (floor detection, levelling, vectorization)
    export    writing it out (2D CAD, 3D meshes, splatting datasets)
    service   serving it over HTTP

`pipeline.py` is the orchestrator that runs ingest -> geometry -> export; the
service layer calls it on a background worker.
"""

__version__ = "2.0.0"
