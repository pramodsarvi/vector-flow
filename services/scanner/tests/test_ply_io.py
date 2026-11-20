"""PLY reading and writing: the format the capture clients and engine share."""

import numpy as np
import pytest

from engine.ingest import ply_io


def test_reads_the_sample_cloud(sample_dir):
    points, colors = ply_io.read_ply(str(sample_dir / "raw_lidar_cloud.ply"))
    assert points.shape[1] == 3
    assert colors.shape == points.shape
    assert len(points) > 1000
    assert np.isfinite(points).all()


@pytest.mark.parametrize("binary", [False, True])
def test_round_trips(tmp_path, binary):
    points = np.array([[0, 0, 0], [1.5, 2.25, -3.75], [10, -10, 0.5]], dtype=np.float32)
    colors = np.array([[255, 0, 0], [0, 255, 0], [10, 20, 30]], dtype=np.uint8)
    path = tmp_path / "cloud.ply"

    ply_io.write_ply(str(path), points, colors, binary=binary)
    read_points, read_colors = ply_io.read_ply(str(path))

    assert np.allclose(points, read_points)
    assert np.array_equal(colors, read_colors)


def test_binary_is_smaller_than_ascii(tmp_path):
    points = np.random.default_rng(0).random((500, 3)).astype(np.float32)
    ascii_path, binary_path = tmp_path / "a.ply", tmp_path / "b.ply"
    ply_io.write_ply(str(ascii_path), points, binary=False)
    ply_io.write_ply(str(binary_path), points, binary=True)
    assert binary_path.stat().st_size < ascii_path.stat().st_size


def test_reads_big_endian_with_float_colors(tmp_path):
    header = (
        b"ply\nformat binary_big_endian 1.0\nelement vertex 2\n"
        b"property double x\nproperty double y\nproperty double z\n"
        b"property float r\nproperty float g\nproperty float b\nend_header\n"
    )
    records = np.array(
        [(1.0, 2.0, 3.0, 1.0, 0.0, 0.0), (4.0, 5.0, 6.0, 0.0, 1.0, 0.0)],
        dtype=[("x", ">f8"), ("y", ">f8"), ("z", ">f8"), ("r", ">f4"), ("g", ">f4"), ("b", ">f4")],
    )
    path = tmp_path / "be.ply"
    path.write_bytes(header + records.tobytes())

    points, colors = ply_io.read_ply(str(path))
    assert np.allclose(points, [[1, 2, 3], [4, 5, 6]])
    # Float colour channels are 0..1 and must be scaled, not truncated.
    assert np.array_equal(colors, [[255, 0, 0], [0, 255, 0]])


def test_cloud_without_color_gets_a_default(tmp_path):
    path = tmp_path / "plain.ply"
    path.write_text(
        "ply\nformat ascii 1.0\nelement vertex 2\n"
        "property float x\nproperty float y\nproperty float z\nend_header\n"
        "0 0 0\n1 1 1\n"
    )
    points, colors = ply_io.read_ply(str(path))
    assert len(points) == 2
    assert colors.shape == (2, 3)


@pytest.mark.parametrize(
    "content, fragment",
    [
        (b"this is not a ply", "missing 'ply' magic"),
        (b"ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\nend_header\n0\n", "missing coordinate"),
        (b"ply\nformat ascii 1.0\nelement vertex 0\nproperty float x\nproperty float y\n"
         b"property float z\nend_header\n", "zero vertices"),
        (b"ply\nformat weird 1.0\nelement vertex 1\nproperty float x\nproperty float y\n"
         b"property float z\nend_header\n", "Unsupported PLY format"),
    ],
)
def test_rejects_malformed_files(tmp_path, content, fragment):
    path = tmp_path / "bad.ply"
    path.write_bytes(content)
    with pytest.raises(ply_io.PlyError) as excinfo:
        ply_io.validate_file(str(path))
    assert fragment in str(excinfo.value)


def test_write_rejects_mismatched_colors(tmp_path):
    with pytest.raises(ValueError, match="does not match"):
        ply_io.write_ply(str(tmp_path / "x.ply"), np.zeros((3, 3)), np.zeros((2, 3)))
