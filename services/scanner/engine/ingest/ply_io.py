"""
Point-cloud IO for PLY files — one implementation shared by the capture
clients and the processing engine.

Reads ASCII and binary (little- and big-endian) PLY, which is what real
sensors emit: ARKit and most LiDAR SDKs write binary because an ASCII cloud of
a few million points is an order of magnitude larger and far slower to parse.
Colors are optional and normalised to uint8 RGB.
"""

from __future__ import annotations

import numpy as np

# PLY scalar type names -> numpy dtype characters.
_PLY_TYPES = {
    "char": "i1", "int8": "i1",
    "uchar": "u1", "uint8": "u1",
    "short": "i2", "int16": "i2",
    "ushort": "u2", "uint16": "u2",
    "int": "i4", "int32": "i4",
    "uint": "u4", "uint32": "u4",
    "float": "f4", "float32": "f4",
    "double": "f8", "float64": "f8",
}

_COLOR_ALIASES = {
    "red": "red", "r": "red", "diffuse_red": "red",
    "green": "green", "g": "green", "diffuse_green": "green",
    "blue": "blue", "b": "blue", "diffuse_blue": "blue",
}

DEFAULT_GREY = 200


class PlyError(ValueError):
    """The file is not a PLY we can read. The message is safe to show a user."""


class _Element:
    __slots__ = ("name", "count", "props", "has_list")

    def __init__(self, name: str, count: int):
        self.name = name
        self.count = count
        self.props: list[tuple[str, str]] = []  # (property name, numpy dtype char)
        self.has_list = False


def parse_header(raw: bytes) -> dict:
    """
    Parse a PLY header from the first bytes of a file.

    Returns {"format", "elements", "header_bytes", "vertex_count"}. Useful on
    its own to validate an upload before committing it to disk.
    """
    end = raw.find(b"end_header")
    if not raw.lstrip().lower().startswith(b"ply"):
        raise PlyError("Not a PLY file: missing 'ply' magic line.")
    if end == -1:
        raise PlyError("Malformed PLY: no 'end_header' found in the first bytes.")
    line_end = raw.find(b"\n", end)
    header_bytes = (line_end + 1) if line_end != -1 else (end + len("end_header"))
    text = raw[:header_bytes].decode("ascii", errors="replace")

    fmt: str | None = None
    elements: list[_Element] = []
    for line in text.splitlines():
        parts = line.split()
        if not parts:
            continue
        key = parts[0].lower()
        if key == "format":
            if len(parts) < 2:
                raise PlyError("Malformed PLY: 'format' line is incomplete.")
            fmt = parts[1].lower()
        elif key == "element":
            if len(parts) < 3:
                raise PlyError("Malformed PLY: 'element' line is incomplete.")
            try:
                elements.append(_Element(parts[1], int(parts[2])))
            except ValueError as exc:
                raise PlyError(f"Malformed PLY: bad element count in '{line.strip()}'.") from exc
        elif key == "property":
            if not elements:
                raise PlyError("Malformed PLY: 'property' before any 'element'.")
            if parts[1].lower() == "list":
                elements[-1].has_list = True
                continue
            dtype = _PLY_TYPES.get(parts[1].lower())
            if dtype is None:
                raise PlyError(f"Unsupported PLY property type '{parts[1]}'.")
            elements[-1].props.append((parts[-1].lower(), dtype))

    if fmt not in ("ascii", "binary_little_endian", "binary_big_endian"):
        raise PlyError(f"Unsupported PLY format '{fmt or 'missing'}'.")

    vertex = next((e for e in elements if e.name.lower() == "vertex"), None)
    if vertex is None:
        raise PlyError("PLY has no 'vertex' element.")
    names = {p[0] for p in vertex.props}
    missing = {"x", "y", "z"} - names
    if missing:
        raise PlyError(f"PLY vertices are missing coordinate properties: {', '.join(sorted(missing))}.")

    return {
        "format": fmt,
        "elements": elements,
        "header_bytes": header_bytes,
        "vertex_count": vertex.count,
    }


def validate_file(path: str, probe_bytes: int = 64 * 1024) -> dict:
    """Header-only check: confirms a file is a readable point cloud without parsing it."""
    with open(path, "rb") as f:
        head = f.read(probe_bytes)
    info = parse_header(head)
    if info["vertex_count"] <= 0:
        raise PlyError("PLY declares zero vertices.")
    return {"format": info["format"], "vertex_count": info["vertex_count"]}


def read_ply(file_path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load a PLY into (points Nx3 float32, colors Nx3 uint8).

    Clouds without color get a neutral grey so downstream code can always
    assume a color array exists.
    """
    with open(file_path, "rb") as f:
        raw = f.read()
    info = parse_header(raw)
    elements: list[_Element] = info["elements"]
    vertex = next(e for e in elements if e.name.lower() == "vertex")

    if info["format"] == "ascii":
        points, colors = _read_ascii(raw[info["header_bytes"]:], vertex)
    else:
        byteorder = "<" if info["format"] == "binary_little_endian" else ">"
        points, colors = _read_binary(raw[info["header_bytes"]:], elements, vertex, byteorder)

    if len(points) == 0:
        raise PlyError("PLY contained no readable vertices.")
    if colors is None:
        colors = np.full((len(points), 3), DEFAULT_GREY, dtype=np.uint8)
    return points, colors


def _color_indices(vertex: _Element) -> tuple[list[int], list[str]] | tuple[None, None]:
    """Locate red/green/blue columns among the vertex properties, whatever they're called."""
    found: dict[str, int] = {}
    scales: dict[str, str] = {}
    for idx, (name, dtype) in enumerate(vertex.props):
        channel = _COLOR_ALIASES.get(name)
        if channel and channel not in found:
            found[channel] = idx
            scales[channel] = dtype
    if len(found) != 3:
        return None, None
    order = ["red", "green", "blue"]
    return [found[c] for c in order], [scales[c] for c in order]


def _to_uint8(values: np.ndarray, dtype: str) -> np.ndarray:
    """Float color channels are 0..1 by convention; integer channels are already 0..255."""
    if dtype.startswith("f"):
        return np.clip(values * 255.0, 0, 255).astype(np.uint8)
    return np.clip(values, 0, 255).astype(np.uint8)


def _read_ascii(body: bytes, vertex: _Element) -> tuple[np.ndarray, np.ndarray | None]:
    names = [p[0] for p in vertex.props]
    xi, yi, zi = names.index("x"), names.index("y"), names.index("z")
    color_idx, color_types = _color_indices(vertex)
    width = len(vertex.props)

    rows: list[list[float]] = []
    for line in body.decode("ascii", errors="ignore").splitlines():
        if len(rows) >= vertex.count:
            break
        parts = line.split()
        if len(parts) < width:
            continue  # blank line, or a later element's data
        try:
            rows.append([float(v) for v in parts[:width]])
        except ValueError:
            continue
    if not rows:
        return np.empty((0, 3), dtype=np.float32), None

    table = np.asarray(rows, dtype=np.float64)
    points = table[:, [xi, yi, zi]].astype(np.float32)
    colors = None
    if color_idx is not None:
        colors = np.stack(
            [_to_uint8(table[:, i], t) for i, t in zip(color_idx, color_types)], axis=1
        )
    return points, colors


def _read_binary(
    body: bytes, elements: list[_Element], vertex: _Element, byteorder: str
) -> tuple[np.ndarray, np.ndarray | None]:
    # Skip fixed-size elements declared before the vertex element.
    offset = 0
    for element in elements:
        if element is vertex:
            break
        if element.has_list:
            raise PlyError(
                f"Unsupported binary PLY: variable-length '{element.name}' element precedes the vertices."
            )
        offset += element.count * np.dtype(
            [(f"f{i}", byteorder + d) for i, (_, d) in enumerate(element.props)]
        ).itemsize

    dtype = np.dtype([(name, byteorder + d) for name, d in vertex.props])
    needed = offset + vertex.count * dtype.itemsize
    if len(body) < needed:
        # Truncated file: read the whole vertices we do have rather than failing.
        available = max(0, (len(body) - offset) // dtype.itemsize)
        if available == 0:
            raise PlyError("Truncated binary PLY: no complete vertex records.")
        count = available
    else:
        count = vertex.count

    table = np.frombuffer(body, dtype=dtype, count=count, offset=offset)
    points = np.stack([table["x"], table["y"], table["z"]], axis=1).astype(np.float32)

    colors = None
    color_idx, color_types = _color_indices(vertex)
    if color_idx is not None:
        names = [vertex.props[i][0] for i in color_idx]
        colors = np.stack(
            [_to_uint8(table[n].astype(np.float64), t) for n, t in zip(names, color_types)], axis=1
        )
    return points, colors


def write_ply(
    file_path: str, points: np.ndarray, colors: np.ndarray | None = None, binary: bool = False
) -> None:
    """
    Write an Nx3 cloud with uint8 RGB. `binary=True` produces a
    binary_little_endian file — much smaller and faster for dense clouds.
    """
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    if colors is None:
        colors = np.full((len(points), 3), DEFAULT_GREY, dtype=np.uint8)
    colors = np.asarray(colors).reshape(-1, 3)
    if len(colors) != len(points):
        raise ValueError(f"colors length {len(colors)} does not match points length {len(points)}")
    colors = np.clip(colors, 0, 255).astype(np.uint8)

    fmt = "binary_little_endian 1.0" if binary else "ascii 1.0"
    header = (
        "ply\n"
        f"format {fmt}\n"
        f"element vertex {len(points)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    )

    if binary:
        dtype = np.dtype(
            [("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")]
        )
        records = np.empty(len(points), dtype=dtype)
        records["x"], records["y"], records["z"] = points[:, 0], points[:, 1], points[:, 2]
        records["red"], records["green"], records["blue"] = colors[:, 0], colors[:, 1], colors[:, 2]
        with open(file_path, "wb") as f:
            f.write(header.encode("ascii"))
            f.write(records.tobytes())
        return

    with open(file_path, "w") as f:
        f.write(header)
        for p, c in zip(points, colors):
            f.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f} {int(c[0])} {int(c[1])} {int(c[2])}\n")
