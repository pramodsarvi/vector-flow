"""
Writing deliverables.

`cad` produces the 2D drawings (SVG, DXF, blueprint PDF), `mesh` the 3D room
shell and its physics colliders, and `splatting` the multi-view dataset an
external Gaussian-splatting trainer consumes.
"""

from .cad import export_to_dxf, export_to_pdf, export_to_svg
from .mesh import export_physics_colliders, export_textured_room_glb, generate_architectural_3d_shell
from .splatting import prepare_3dgs_dataset

__all__ = [
    "export_to_svg", "export_to_dxf", "export_to_pdf",
    "generate_architectural_3d_shell", "export_textured_room_glb", "export_physics_colliders",
    "prepare_3dgs_dataset",
]
