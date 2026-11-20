"""
Reading capture data.

Everything that turns bytes on disk into arrays the geometry stage can use,
and the validation that rejects a bad bundle before any work is done.
"""

from .bundle import ScanBundle, ScanInputError, open_bundle
from .ply_io import PlyError, read_ply, validate_file, write_ply

__all__ = [
    "PlyError", "read_ply", "write_ply", "validate_file",
    "ScanBundle", "ScanInputError", "open_bundle",
]
