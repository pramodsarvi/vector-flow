"""
3D World Reconstruction & Mesher Module (Production Grade)
1. Extrudes 2D architectural floor plan into 3D structural shell (walls, floor, ceiling, openings)
   with watertight cutouts for doors, windows, sills, and headers.
2. Arbitrary polygon triangulation for floor & ceiling (supports L-shaped, T-shaped, multi-wall rooms).
3. Ingests detected furniture footprints and creates 3D proxy geometries.
4. Exports clean .OBJ / .GLB models and physics collision geometry for Rapier / PhysX.
"""

import json
import math
import os

import trimesh
from shapely.geometry import Polygon as ShapelyPolygon

DEFAULT_WALL_COLOR = [230, 230, 235, 255]


def openings_on_wall_segment(wall: dict, openings: list):
    """
    Computes opening intervals [a, b] in meters along the wall segment from start to end.
    """
    sx, sz = wall["start"]["x"], wall["start"]["z"]
    ex, ez = wall["end"]["x"], wall["end"]["z"]
    dx, dz = ex - sx, ez - sz
    length = math.hypot(dx, dz)
    if length < 1e-4:
        return []

    ux, uz = dx / length, dz / length
    # Distance of point p from the wall line and along the wall line
    def along(p): return (p["x"] - sx) * ux + (p["z"] - sz) * uz
    def off(p): return abs(-(p["x"] - sx) * uz + (p["z"] - sz) * ux)

    result = []
    for op in openings:
        if off(op["start"]) < 0.30 and off(op["end"]) < 0.30:
            a = max(0.0, min(along(op["start"]), along(op["end"])))
            b = min(length, max(along(op["start"]), along(op["end"])))
            if b - a > 0.05:
                result.append({"op": op, "a": a, "b": b})

    result.sort(key=lambda item: item["a"])
    return result

def triangulate_2d_polygon(polygon_vertices: list):
    """
    Triangulates an arbitrary 2D polygon (convex or concave, e.g. L-shaped)
    into a list of 2D triangle vertex indices.
    """
    poly_pts = [(p["x"], p["z"]) for p in polygon_vertices]
    shapely_poly = ShapelyPolygon(poly_pts)

    # Use trimesh triangulation
    vertices_2d, faces = trimesh.creation.triangulate_polygon(shapely_poly)
    return vertices_2d, faces

def generate_architectural_3d_shell(
    floorplan: dict,
    wall_thickness: float = 0.15,
    ceiling_height: float = 2.8,
    output_obj: str = None
):
    """
    Constructs a metric-accurate 3D world mesh from the vectorized floorplan,
    including cutouts for doors and windows in the walls, arbitrary polygon triangulation,
    and 3D furniture proxy volumes.
    """
    room = floorplan["room"]
    polygon = room["polygon"]
    openings = room.get("openings", [])
    walls = room.get("walls", [])
    furniture = room.get("furniture_footprints", [])

    scene = trimesh.Scene()

    # 1. Floor & Ceiling Meshes (Arbitrary polygon triangulation)
    try:
        pts_2d, tri_faces = triangulate_2d_polygon(polygon)
        v_floor = [[p[0], 0.0, p[1]] for p in pts_2d]
        v_ceil = [[p[0], ceiling_height, p[1]] for p in pts_2d]
        f_floor = tri_faces
        # Invert face winding for ceiling (normal facing downwards into the room)
        f_ceil = [[f[0], f[2], f[1]] for f in tri_faces]
    except Exception:
        # Fallback to simple fan triangulation
        n = len(polygon)
        v_floor = [[p["x"], 0.0, p["z"]] for p in polygon]
        v_ceil = [[p["x"], ceiling_height, p["z"]] for p in polygon]
        f_floor = [[0, i, i + 1] for i in range(1, n - 1)]
        f_ceil = [[0, i + 1, i] for i in range(1, n - 1)]

    floor_mesh = trimesh.Trimesh(vertices=v_floor, faces=f_floor, process=True)
    floor_mesh.visual.vertex_colors = [140, 95, 55, 255] # Wood floor
    scene.add_geometry(floor_mesh, node_name="floor")

    ceil_mesh = trimesh.Trimesh(vertices=v_ceil, faces=f_ceil, process=True)
    ceil_mesh.visual.vertex_colors = [240, 240, 245, 120] # Ceiling
    scene.add_geometry(ceil_mesh, node_name="ceiling")

    # 2. Wall meshes with Door and Window cutouts
    wall_meshes = []
    glass_meshes = []

    for w in walls:
        sx, sz = w["start"]["x"], w["start"]["z"]
        ex, ez = w["end"]["x"], w["end"]["z"]
        dx, dz = ex - sx, ez - sz
        wall_len = math.hypot(dx, dz)
        if wall_len < 1e-4:
            continue
        ux, uz = dx / wall_len, dz / wall_len
        rot_y = -math.atan2(dz, dx)

        openings_list = openings_on_wall_segment(w, openings)

        # The loop variables are bound as defaults rather than captured: the
        # closure is only called within this iteration today, but binding makes
        # that explicit and survives any later refactor that defers the call.
        def make_wall_box(a, b, y0, y1, color=None, is_glass=False,
                          sx=sx, sz=sz, ux=ux, uz=uz, rot_y=rot_y):
            color = DEFAULT_WALL_COLOR if color is None else color
            if (b - a) < 0.01 or (y1 - y0) < 0.01:
                return
            box_w = b - a
            box_h = y1 - y0
            box_d = 0.04 if is_glass else wall_thickness
            m = trimesh.creation.box(extents=[box_w, box_h, box_d])
            # Center of box along wall
            mid = (a + b) / 2.0
            cx = sx + ux * mid
            cy = (y0 + y1) / 2.0
            cz = sz + uz * mid
            # Rotate around Y axis and translate
            m.apply_transform(trimesh.transformations.rotation_matrix(rot_y, [0, 1, 0]))
            m.apply_translation([cx, cy, cz])
            m.visual.vertex_colors = color
            if is_glass:
                glass_meshes.append(m)
            else:
                wall_meshes.append(m)

        cursor = 0.0
        for item in openings_list:
            op = item["op"]
            a = item["a"]
            b = item["b"]

            # Wall piece before opening
            make_wall_box(cursor, a, 0.0, ceiling_height)

            sill = op.get("sill_height_m", 0.0) or 0.0
            op_h = op.get("height_m", 2.1) or 2.1
            top = min(ceiling_height, sill + op_h)

            # Piece below sill (for windows)
            if sill > 0.01:
                make_wall_box(a, b, 0.0, sill)

            # Header / lintel piece above opening
            if top < ceiling_height - 0.01:
                make_wall_box(a, b, top, ceiling_height)

            # Glass pane for window
            if op.get("type") == "window":
                make_wall_box(a, b, sill, top, color=[56, 189, 248, 120], is_glass=True)

            cursor = max(cursor, b)

        # Wall piece after last opening
        make_wall_box(cursor, wall_len, 0.0, ceiling_height)

    if wall_meshes:
        combined_walls = trimesh.util.concatenate(wall_meshes)
        scene.add_geometry(combined_walls, node_name="walls")
    if glass_meshes:
        combined_glass = trimesh.util.concatenate(glass_meshes)
        scene.add_geometry(combined_glass, node_name="windows")

    # 3. Furniture 3D Proxies
    for idx, fn in enumerate(furniture):
        bbox = fn["bbox"]
        cx = bbox["center_x"]
        cz = bbox["center_z"]
        w_m = bbox["width_m"]
        d_m = bbox["depth_m"]
        h_m = bbox["height_m"]

        box = trimesh.creation.box(extents=[w_m, h_m, d_m])
        box.apply_translation([cx, h_m / 2.0, cz])

        if "Sofa" in fn["type"]:
            box.visual.vertex_colors = [70, 100, 150, 255] # Navy
        elif "Table" in fn["type"]:
            box.visual.vertex_colors = [160, 110, 65, 255] # Wood brown
        else:
            box.visual.vertex_colors = [120, 130, 140, 255]

        scene.add_geometry(box, node_name=f"furniture_{idx}")

    # Export combined mesh
    try:
        combined_mesh = scene.to_geometry()
    except AttributeError:
        combined_mesh = scene.dump(concatenate=True)

    if output_obj:
        os.makedirs(os.path.dirname(os.path.abspath(output_obj)), exist_ok=True)
        combined_mesh.export(output_obj)

    return combined_mesh

def export_textured_room_glb(mesh: trimesh.Trimesh, output_glb: str):
    """
    Exports a GLTF/GLB binary file ready for WebGL/Three.js rendering.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_glb)), exist_ok=True)
    mesh.export(output_glb)
    return output_glb

def export_physics_colliders(floorplan: dict, output_json: str):
    """
    Exports bounding boxes and planar colliders for first-person walking physics.
    """
    colliders = []
    room = floorplan["room"]
    ceiling_h = floorplan.get("ceiling_height_m", 2.8)

    # Floor plane
    colliders.append({
        "type": "plane",
        "normal": [0, 1, 0],
        "offset": 0.0,
        "description": "Floor ground collider"
    })

    # Wall segment colliders (taking into account door openings where a user can walk through)
    openings = room.get("openings", [])
    doors = [op for op in openings if op.get("type") == "door"]

    for w in room.get("walls", []):
        wall_doors = openings_on_wall_segment(w, doors)
        sx, sz = w["start"]["x"], w["start"]["z"]
        ex, ez = w["end"]["x"], w["end"]["z"]
        dx, dz = ex - sx, ez - sz
        w_len = math.hypot(dx, dz)
        if w_len < 1e-4:
            continue
        ux, uz = dx / w_len, dz / w_len

        cursor = 0.0
        for d in wall_doors:
            a, b = d["a"], d["b"]
            if a - cursor > 0.10:
                colliders.append({
                    "type": "wall_segment",
                    "start": [sx + ux * cursor, 0, sz + uz * cursor],
                    "end": [sx + ux * a, 0, sz + uz * a],
                    "height": ceiling_h
                })
            cursor = max(cursor, b)
        if w_len - cursor > 0.10:
            colliders.append({
                "type": "wall_segment",
                "start": [sx + ux * cursor, 0, sz + uz * cursor],
                "end": [ex, 0, ez],
                "height": ceiling_h
            })

    # Furniture obstacle colliders
    for fn in room.get("furniture_footprints", []):
        b = fn["bbox"]
        colliders.append({
            "type": "box",
            "center": [b["center_x"], b["height_m"] / 2.0, b["center_z"]],
            "extents": [b["width_m"], b["height_m"], b["depth_m"]],
            "label": fn["type"]
        })

    os.makedirs(os.path.dirname(os.path.abspath(output_json)), exist_ok=True)
    with open(output_json, "w") as f:
        json.dump({"colliders": colliders}, f, indent=2)

    return output_json
