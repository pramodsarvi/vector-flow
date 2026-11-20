"""
CAD, Vector & Blueprint Exporter Module
Exports structured architectural floor design to:
1. Scalable Vector Graphics (SVG) - interactive web layout with furniture & openings
2. DXF (AutoCAD standard layers: WALLS, DOORS, WINDOWS, FURNITURE, DIMENSIONS)
3. Printable Architectural Blueprint PDF
4. Structured Floorplan JSON
"""

import math
import os

import ezdxf
import matplotlib

matplotlib.use("Agg")  # headless backend: required when called from API worker threads (macOS GUI backend crashes off-main-thread)
import matplotlib.patches as patches
import matplotlib.pyplot as plt


def export_to_svg(floorplan: dict, output_file: str, scale: float = 70.0, padding: float = 60.0):
    """
    Renders an architectural 2D floor plan to SVG including walls, openings, and furniture.
    """
    room = floorplan["room"]
    polygon = room["polygon"]
    openings = room.get("openings", [])
    walls = room.get("walls", [])
    furniture = room.get("furniture_footprints", [])

    xs = [p["x"] for p in polygon]
    zs = [p["z"] for p in polygon]
    min_x, max_x = min(xs), max(xs)
    min_z, max_z = min(zs), max(zs)

    width_m = max_x - min_x
    depth_m = max_z - min_z

    svg_width = int(width_m * scale + padding * 2)
    svg_height = int(depth_m * scale + padding * 2)

    def to_svg_coord(x, z):
        sx = padding + (x - min_x) * scale
        sy = padding + (z - min_z) * scale
        return sx, sy

    svg = []
    svg.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {svg_width} {svg_height}" width="{svg_width}" height="{svg_height}">')
    svg.append('<defs>')
    svg.append('''
        <pattern id="grid" width="20" height="20" patternUnits="userSpaceOnUse">
            <path d="M 20 0 L 0 0 0 20" fill="none" stroke="#f0f2f5" stroke-width="1"/>
        </pattern>
        <pattern id="floor-tile" width="40" height="40" patternUnits="userSpaceOnUse">
            <path d="M 40 0 L 0 0 0 40" fill="none" stroke="#e8ecef" stroke-width="0.8"/>
        </pattern>
    ''')
    svg.append('</defs>')

    # Background
    svg.append(f'<rect width="{svg_width}" height="{svg_height}" fill="#ffffff"/>')
    svg.append(f'<rect width="{svg_width}" height="{svg_height}" fill="url(#grid)"/>')

    # Floor fill polygon
    poly_pts = " ".join([f"{to_svg_coord(p['x'], p['z'])[0]},{to_svg_coord(p['x'], p['z'])[1]}" for p in polygon])
    svg.append(f'<polygon points="{poly_pts}" fill="#fafbfc" stroke="#cbd5e1" stroke-width="1" />')
    svg.append(f'<polygon points="{poly_pts}" fill="url(#floor-tile)" />')

    # Draw Furniture Footprints
    for fn in furniture:
        bbox = fn["bbox"]
        fx, fz = to_svg_coord(bbox["center_x"] - bbox["width_m"] / 2.0, bbox["center_z"] - bbox["depth_m"] / 2.0)
        fw = bbox["width_m"] * scale
        fd = bbox["depth_m"] * scale

        # Rounded furniture rectangle
        svg.append(f'<rect x="{fx:.1f}" y="{fz:.1f}" width="{fw:.1f}" height="{fd:.1f}" rx="6" fill="#e0e7ff" stroke="#6366f1" stroke-width="1.5" stroke-dasharray="3,3" opacity="0.9"/>')
        # Furniture label
        svg.append(f'<text x="{fx + fw/2:.1f}" y="{fz + fd/2 + 3:.1f}" font-family="sans-serif" font-size="10" font-weight="600" fill="#4338ca" text-anchor="middle">{fn["type"]}</text>')

    # Draw Wall Lines
    for w in walls:
        x1, y1 = to_svg_coord(w["start"]["x"], w["start"]["z"])
        x2, y2 = to_svg_coord(w["end"]["x"], w["end"]["z"])
        svg.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="#1e293b" stroke-width="8" stroke-linecap="square"/>')

    # Draw Openings (Doors & Windows)
    for op in openings:
        sx, sy = to_svg_coord(op["start"]["x"], op["start"]["z"])
        ex, ey = to_svg_coord(op["end"]["x"], op["end"]["z"])

        # Wall gap clearance
        svg.append(f'<line x1="{sx:.1f}" y1="{sy:.1f}" x2="{ex:.1f}" y2="{ey:.1f}" stroke="#ffffff" stroke-width="9"/>')

        if op["type"] == "door":
            dx = ex - sx
            dy = ey - sy
            w_len = math.sqrt(dx*dx + dy*dy)
            svg.append(f'<line x1="{sx:.1f}" y1="{sy:.1f}" x2="{sx + dx*0.2:.1f}" y2="{sy + dy*0.2:.1f}" stroke="#2563eb" stroke-width="3"/>')
            svg.append(f'<path d="M {ex:.1f} {ey:.1f} A {w_len:.1f} {w_len:.1f} 0 0 0 {sx:.1f} {sy + w_len:.1f}" fill="none" stroke="#3b82f6" stroke-width="1.5" stroke-dasharray="4,4"/>')
            svg.append(f'<line x1="{sx:.1f}" y1="{sy:.1f}" x2="{sx:.1f}" y2="{sy + w_len:.1f}" stroke="#1d4ed8" stroke-width="2.5"/>')
            mid_x = (sx + ex) / 2
            mid_y = (sy + ey) / 2
            svg.append(f'<text x="{mid_x:.1f}" y="{mid_y - 8:.1f}" font-family="sans-serif" font-size="10" fill="#2563eb" font-weight="bold" text-anchor="middle">DOOR ({op["width_m"]}m)</text>')

        elif op["type"] == "window":
            svg.append(f'<line x1="{sx:.1f}" y1="{sy:.1f}" x2="{ex:.1f}" y2="{ey:.1f}" stroke="#0284c7" stroke-width="3"/>')
            svg.append(f'<line x1="{sx:.1f}" y1="{sy-3:.1f}" x2="{ex:.1f}" y2="{ey-3:.1f}" stroke="#38bdf8" stroke-width="1.5"/>')
            svg.append(f'<line x1="{sx:.1f}" y1="{sy+3:.1f}" x2="{ex:.1f}" y2="{ey+3:.1f}" stroke="#38bdf8" stroke-width="1.5"/>')
            mid_x = (sx + ex) / 2
            mid_y = (sy + ey) / 2
            svg.append(f'<text x="{mid_x:.1f}" y="{mid_y - 10:.1f}" font-family="sans-serif" font-size="10" fill="#0284c7" font-weight="bold" text-anchor="middle">WINDOW ({op["width_m"]}m)</text>')

    # Dimension Callouts
    for w in walls:
        x1, y1 = to_svg_coord(w["start"]["x"], w["start"]["z"])
        x2, y2 = to_svg_coord(w["end"]["x"], w["end"]["z"])
        mx = (x1 + x2) / 2
        my = (y1 + y2) / 2

        dx = x2 - x1
        dy = y2 - y1
        length = math.sqrt(dx*dx + dy*dy)
        if length > 0:
            nx = -dy / length * 22
            ny = dx / length * 22
            label = f"{w['length_m']}m ({w['length_ft']}ft)"
            svg.append(f'<text x="{mx + nx:.1f}" y="{my + ny + 4:.1f}" font-family="sans-serif" font-size="11" fill="#475569" font-weight="bold" text-anchor="middle">{label}</text>')

    # Room Center Badge
    center_x = padding + (width_m / 2) * scale
    center_y = padding + (depth_m / 2) * scale
    svg.append(f'''
        <g transform="translate({center_x}, {center_y})">
            <rect x="-105" y="-35" width="210" height="70" rx="8" fill="#ffffff" stroke="#e2e8f0" stroke-width="1.5" filter="drop-shadow(0 2px 4px rgba(0,0,0,0.05))"/>
            <text x="0" y="-12" font-family="sans-serif" font-size="13" font-weight="bold" fill="#0f172a" text-anchor="middle">{room["name"]}</text>
            <text x="0" y="8" font-family="sans-serif" font-size="12" font-weight="600" fill="#2563eb" text-anchor="middle">{room["area_sqm"]} m² ({room["area_sqft"]} sq ft)</text>
            <text x="0" y="24" font-family="sans-serif" font-size="10" fill="#64748b" text-anchor="middle">Perimeter: {room["perimeter_m"]}m</text>
        </g>
    ''')

    svg.append('</svg>')

    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    with open(output_file, "w") as f:
        f.write("\n".join(svg))
    return output_file

def export_to_dxf(floorplan: dict, output_file: str):
    """
    Exports to Autodesk DXF standard with distinct architectural layers.
    """
    doc = ezdxf.new("R2010")
    msp = doc.modelspace()

    # Create CAD Layers
    doc.layers.add(name="WALLS", color=7)         # White / Black
    doc.layers.add(name="DOORS", color=5)         # Blue
    doc.layers.add(name="WINDOWS", color=4)       # Cyan
    doc.layers.add(name="FURNITURE", color=1)     # Red
    doc.layers.add(name="DIMENSIONS", color=2)    # Yellow
    doc.layers.add(name="ANNOTATIONS", color=3)   # Green

    room = floorplan["room"]
    polygon = room["polygon"]
    openings = room.get("openings", [])
    walls = room.get("walls", [])
    furniture = room.get("furniture_footprints", [])

    # 1. Walls
    for w in walls:
        msp.add_line(
            (w["start"]["x"], w["start"]["z"]),
            (w["end"]["x"], w["end"]["z"]),
            dxfattribs={"layer": "WALLS", "lineweight": 50}
        )

    # 2. Openings
    for op in openings:
        sx, sz = op["start"]["x"], op["start"]["z"]
        ex, ez = op["end"]["x"], op["end"]["z"]
        layer = "DOORS" if op["type"] == "door" else "WINDOWS"
        msp.add_line((sx, sz), (ex, ez), dxfattribs={"layer": layer, "lineweight": 25})

    # 3. Furniture Footprints
    for fn in furniture:
        bbox = fn["bbox"]
        cx, cz = bbox["center_x"], bbox["center_z"]
        hw, hd = bbox["width_m"] / 2.0, bbox["depth_m"] / 2.0

        # Rectangle polyline
        pts = [
            (cx - hw, cz - hd),
            (cx + hw, cz - hd),
            (cx + hw, cz + hd),
            (cx - hw, cz + hd),
            (cx - hw, cz - hd)
        ]
        msp.add_lwpolyline(pts, dxfattribs={"layer": "FURNITURE", "lineweight": 18})
        msp.add_text(fn["type"], dxfattribs={"layer": "FURNITURE", "height": 0.12}).set_placement((cx - hw * 0.8, cz))

    # 4. Text dimensions
    for w in walls:
        mx = (w["start"]["x"] + w["end"]["x"]) / 2
        mz = (w["start"]["z"] + w["end"]["z"]) / 2
        msp.add_text(
            f"{w['length_m']}m",
            dxfattribs={"layer": "DIMENSIONS", "height": 0.15}
        ).set_placement((mx, mz))

    # Center Room Tag
    xs = [p["x"] for p in polygon]
    zs = [p["z"] for p in polygon]
    cx = (min(xs) + max(xs)) / 2
    cz = (min(zs) + max(zs)) / 2
    msp.add_text(
        f"{room['name']} - {room['area_sqm']} sq.m",
        dxfattribs={"layer": "ANNOTATIONS", "height": 0.25}
    ).set_placement((cx, cz))

    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    doc.saveas(output_file)
    return output_file

def export_to_pdf(floorplan: dict, output_file: str):
    """
    Renders high-resolution printable architectural blueprint PDF.
    """
    room = floorplan["room"]
    polygon = room["polygon"]
    openings = room.get("openings", [])
    walls = room.get("walls", [])
    furniture = room.get("furniture_footprints", [])

    xs = [p["x"] for p in polygon]
    zs = [p["z"] for p in polygon]
    min_x, max_x = min(xs), max(xs)
    min_z, max_z = min(zs), max(zs)

    fig, ax = plt.subplots(figsize=(10, 8), dpi=200)
    ax.set_facecolor('#f8fafc')

    # Grid
    ax.grid(True, linestyle='--', color='#e2e8f0', alpha=0.7)

    # Floor polygon
    poly_coords = [[p["x"], p["z"]] for p in polygon]
    poly_patch = patches.Polygon(poly_coords, closed=True, facecolor='#ffffff', edgecolor='#94a3b8', linewidth=1.5)
    ax.add_patch(poly_patch)

    # Furniture
    for fn in furniture:
        bbox = fn["bbox"]
        fx = bbox["center_x"] - bbox["width_m"] / 2.0
        fz = bbox["center_z"] - bbox["depth_m"] / 2.0
        rect = patches.Rectangle((fx, fz), bbox["width_m"], bbox["depth_m"],
                                 facecolor='#e0e7ff', edgecolor='#6366f1', linestyle='--', linewidth=1.2)
        ax.add_patch(rect)
        ax.text(bbox["center_x"], bbox["center_z"], fn["type"],
                color='#4338ca', fontsize=8, ha='center', va='center', weight='bold')

    # Walls
    for w in walls:
        ax.plot([w["start"]["x"], w["end"]["x"]], [w["start"]["z"], w["end"]["z"]],
                color='#0f172a', linewidth=4.0, solid_capstyle='projecting')
        # Dimension label
        mx = (w["start"]["x"] + w["end"]["x"]) / 2
        mz = (w["start"]["z"] + w["end"]["z"]) / 2
        ax.text(mx, mz, f"{w['length_m']}m", color='#334155', fontsize=9,
                ha='center', va='center', backgroundcolor='#ffffff')

    # Openings
    for op in openings:
        color = '#2563eb' if op["type"] == "door" else '#0284c7'
        ax.plot([op["start"]["x"], op["end"]["x"]], [op["start"]["z"], op["end"]["z"]],
                color=color, linewidth=5.5)

    # Title & Metadata Block
    ax.text(0.03, 0.95, f"ARCHITECTURAL FLOOR SCAN: {room['name']}", transform=ax.transAxes,
            fontsize=12, weight='bold', color='#0f172a')
    ax.text(0.03, 0.91, f"Area: {room['area_sqm']} m² ({room['area_sqft']} sq ft) | Perimeter: {room['perimeter_m']} m",
            transform=ax.transAxes, fontsize=10, color='#2563eb')
    ax.text(0.03, 0.88, "Captured with Camera + LiDAR Sensor Fusion", transform=ax.transAxes,
            fontsize=8, color='#64748b')

    margin = 1.0
    ax.set_xlim(min_x - margin, max_x + margin)
    ax.set_ylim(max_z + margin, min_z - margin) # Invert Z for floorplan orientation
    ax.set_aspect('equal')
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Z (meters)')

    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    plt.savefig(output_file, bbox_inches='tight')
    plt.close(fig)
    return output_file
