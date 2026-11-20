"""Catalogue generation/maps and surface geometry (numpy only)."""
import json
import shutil

import numpy as np
import pytest
from conftest import ROOT, UP, H, K, W, rays

import structsync.catalog as C
from structsync.surfaces import Plane, Surface, align_horizontal_axes, compute_uv, drop_ragged


@pytest.fixture(scope="module")
def demo_catalog(tmp_path_factory):
    root = tmp_path_factory.mktemp("catalog")
    old = C.ROOT
    C.ROOT = root
    try:
        C.build_demo_catalog()
    finally:
        C.ROOT = old
    return root


def test_demo_catalogue_matches_committed_catalogue(demo_catalog):
    """Generators are deterministic and adding height maps didn't change any albedo."""
    entries = json.loads((demo_catalog / "catalog.json").read_text())
    assert entries == json.loads((ROOT / "tests" / "fixtures" / "catalog" / "catalog.json").read_text())
    for e in entries:
        assert (demo_catalog / e["id"] / "albedo.png").read_bytes() == \
            (ROOT / "tests" / "fixtures" / "catalog" / e["id"] / "albedo.png").read_bytes(), e["id"]


def test_demo_textures_are_seamless(demo_catalog):
    for sid in ("CT-CAL-01", "CT-OAK-04", "BS-SUB-01"):
        a = np.asarray(C.Image.open(demo_catalog / sid / "albedo.png"), np.float32)
        inner = np.abs(np.diff(a, axis=1)).mean()
        wrap = np.abs(a[:, 0] - a[:, -1]).mean()
        assert wrap < 3 * inner + 2, sid


def test_write_demo_maps_only_creates_missing_and_skips_replaced(tmp_path, demo_catalog):
    root = tmp_path / "cat"
    shutil.copytree(demo_catalog, root)
    shutil.copy(root / "CT-CAL-01" / "albedo.png", root / "FL-OAK-01" / "albedo.png")   # "supplier" texture
    before = sorted(p.name for p in root.iterdir())
    wrote = C.write_demo_maps(root)
    assert set(wrote) == {"CT-OAK-04", "CB-WAL-04", "BS-SUB-01", "BS-GRN-02",
                         "CL-OAK-02", "CL-ACO-03"}
    assert not (root / "FL-OAK-01" / "normal.png").exists()
    assert C.write_demo_maps(root) == []
    assert sorted(p.name for p in root.iterdir()) == before          # catalog.json / embeddings untouched
    sku = next(s for s in _load(root) if s.id == "BS-SUB-01")
    n = sku.normal
    assert n.shape[2] == 3 and np.allclose(np.linalg.norm(n, axis=-1), 1, atol=0.02)
    assert 0 <= sku.roughness.min() and sku.roughness.max() <= 1


def _load(root):
    data = json.loads((root / "catalog.json").read_text())
    return [C.SKU(path=root / d["id"], **d) for d in data]


def test_height_to_normal_convention():
    """OpenGL convention (+G = texture up = towards row 0). A surface that rises towards the top of the
    texture is a slope facing *down*, so its normal must have a negative G component."""
    rises_up = np.tile(np.linspace(1, 0, 64, dtype=np.float32)[:, None], (1, 64))
    n = C.height_to_normal(rises_up)
    assert n[32, 32, 1] < 0 and abs(n[32, 32, 0]) < 1e-6
    n = C.height_to_normal(rises_up[::-1].copy())                # rises downward → faces up
    assert n[32, 32, 1] > 0


def test_compute_uv_is_metric():
    m = np.zeros((H, W), bool)
    m[:, :] = True
    p = Plane(np.array([0.0, 0, 1]), 3.0, np.array([1.0, 0, 0]), np.array([0.0, 1, 0]), m, False)
    s = Surface("wall", m, [p])
    compute_uv(s, K, (H, W))
    # 30 px at 3 m with f = 300 px → 0.3 m
    assert np.isclose(s.uv[120, 190, 0] - s.uv[120, 160, 0], 0.3, atol=1e-4)
    assert np.isclose(s.footprint[120, 160], 3.0 / 300, rtol=1e-3)


def test_counter_grain_follows_the_wall():
    r = rays()
    m = r[..., 1] > 0.05
    counter = Plane(np.array([0.0, 1, 0]), 1.0, np.array([1.0, 0, 0]), np.array([0.0, 0, 1]), m, True)
    wall_n = np.array([np.sin(0.4), 0, np.cos(0.4)])          # wall rotated 23° from the image plane
    wall = Plane(wall_n, 3.0, np.zeros(3), np.zeros(3), ~m, False)
    surfs = {"countertop": Surface("countertop", m, [counter]), "backsplash": Surface("backsplash", ~m, [wall])}
    align_horizontal_axes(surfs, UP)
    assert abs(counter.eu @ wall_n) < 1e-6 and abs(counter.eu @ counter.n) < 1e-6


def test_drop_ragged_keeps_panels_drops_speckle():
    m = np.zeros((200, 200), bool)
    m[20:80, 20:80] = True                                       # a door panel
    speck = np.zeros_like(m)                                     # SAM-style partial mask: a ragged comb
    speck[110:116, 110:190] = True
    for x in range(110, 190, 12):
        speck[110:190, x:x + 4] = True
    out = drop_ragged(m | speck)
    assert out[20:80, 20:80].all()
    assert out[110:190, 110:190].sum() < 0.2 * speck[110:190, 110:190].sum()
