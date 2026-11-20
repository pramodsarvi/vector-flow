"""Renderer behaviour on synthetic planes (numpy only)."""
import numpy as np
import pytest
from conftest import UP, H, K, W
from PIL import Image

import structsync.render as R
from structsync.catalog import SKU, encode_normal, height_to_normal
from structsync.surfaces import Plane, Surface, compute_uv

xx = np.mgrid[0:H, 0:W][1]


def _surface(n, d, eu, ev, mask):
    p = Plane(np.array(n, float), float(d), np.array(eu, float), np.array(ev, float), mask, False)
    s = Surface("wall", mask, [p])
    compute_uv(s, K, (H, W))
    return s


FRONT = dict(n=[0, 0, 1], d=3.0, eu=[1, 0, 0], ev=[0, 1, 0], mask=xx >= 120)     # facing the camera
SIDE = dict(n=[-1, 0, 0], d=1.0, eu=[0, 0, 1], ev=[0, 1, 0], mask=xx < 100)      # left wall, x = −1


@pytest.fixture
def scene():
    rng = np.random.default_rng(0)
    import cv2
    img = np.clip(0.55 + 0.25 * cv2.GaussianBlur(rng.random((H, W, 3)).astype(np.float32), (0, 0), 6), 0, 1)
    img[30:60, 200:260] = 0.97
    return img


@pytest.fixture
def sku(tmp_path):
    d = tmp_path / "SKU1"
    d.mkdir()
    rng = np.random.default_rng(1)
    Image.fromarray((rng.random((256, 256, 3)) * 255).astype(np.uint8)).save(d / "albedo.png")
    return SKU(id="SKU1", name="t", category="wall", vendor="v", price=1, tile_w=0.5, tile_h=0.5, gloss=0.5,
               path=d)


@pytest.mark.parametrize("geom", [FRONT, SIDE], ids=["front", "side"])
def test_defaults_deterministic_and_maps_noop_without_files(scene, sku, geom):
    s = _surface(**geom)
    a, ea, _ = R.project_materials(scene, {"wall": s}, {"wall": sku}, up=UP)
    b, eb, _ = R.project_materials(scene, {"wall": s}, {"wall": sku}, up=UP)
    c, _, _ = R.project_materials(scene, {"wall": s}, {"wall": sku}, up=UP, material_maps=True)
    assert np.array_equal(a, b) and np.array_equal(ea, eb)
    assert np.array_equal(a, c)


def test_only_edited_pixels_change(scene, sku):
    s = _surface(**FRONT)
    out, edited, _ = R.project_materials(scene, {"wall": s}, {"wall": sku}, up=UP)
    far = ~np.pad(edited, 3)[:-6, 3:-3] & ~edited & ~np.pad(edited, 3)[6:, 3:-3]
    far &= ~np.roll(edited, 3, 1) & ~np.roll(edited, -3, 1)
    assert np.abs(out - scene)[far].max() < 1e-6


@pytest.mark.parametrize("geom", [FRONT, SIDE], ids=["front", "side"])
def test_flat_normal_map_is_identity(scene, sku, geom):
    flat = np.zeros((256, 256, 3), np.uint8)
    flat[..., :2], flat[..., 2] = 128, 255
    Image.fromarray(flat).save(sku.path / "normal.png")
    s = _surface(**geom)
    a, _, _ = R.project_materials(scene, {"wall": s}, {"wall": sku}, up=UP)
    c, _, _ = R.project_materials(scene, {"wall": s}, {"wall": sku}, up=UP, material_maps=True)
    assert np.abs(a - c).max() < 1e-6


@pytest.mark.parametrize("geom", [FRONT, SIDE], ids=["front", "side"])
def test_relief_brightens_up_facing_slopes(sku, geom):
    hgt = np.sin(2 * np.pi * np.arange(256) / 64.0)[:, None].repeat(256, 1).astype(np.float32)
    Image.fromarray(encode_normal(height_to_normal(hgt))).save(sku.path / "normal.png")
    s = _surface(**geom)
    micro, spec = R.relief_shading(sku, s, UP)
    N = R.sample_map(sku.normal, 0.5, 0.5, s)
    m = s.mask & (s.footprint > 0)
    assert np.corrcoef(micro[m] - 1, N[m][:, 1])[0, 1] > 0.9
    assert spec is None


def test_roughness_scales_highlights(sku):
    s = _surface(**FRONT)
    m = s.mask & (s.footprint > 0)
    Image.fromarray(np.full((256, 256), 128, np.uint8)).save(sku.path / "roughness.png")
    assert np.allclose(R.relief_shading(sku, s, UP)[1][m], 1.0, atol=0.005)
    Image.fromarray(np.zeros((256, 256), np.uint8)).save(sku.path / "roughness.png")
    assert np.allclose(R.relief_shading(sku, s, UP)[1][m], 2.0)


def test_neighbour_tint_ignores_coloured_walls_and_is_capped(scene):
    s = _surface(**FRONT)
    lin = R.srgb_to_lin(scene).astype(np.float32)
    g = lin.reshape(-1, 3).mean(0)
    g = (g / g.mean()) ** 0.5
    none = np.zeros((H, W), bool)
    red = lin.copy()
    red[:, 100:120] = [0.6, 0.05, 0.05]
    assert np.allclose(R.neighbour_tint(red, s.mask, none, g) / g, 1, atol=0.0601)
    warm = lin.copy()
    warm[:, :120] *= np.array([1.25, 1.0, 0.8], np.float32)
    t = R.neighbour_tint(warm, s.mask, none, g)
    assert t[0] > g[0] and t[2] < g[2] and np.abs(t / g - 1).max() <= 0.0601


def test_grain_only_touches_edited_pixels_and_is_deterministic(scene):
    rng = np.random.default_rng(3)
    noisy = np.clip(scene + 0.01 * rng.standard_normal(scene.shape).astype(np.float32), 0, 1)
    edited = np.zeros((H, W), bool)
    edited[60:200, 150:300] = True
    clean = noisy.copy()
    clean[edited] = 0.5
    a, info = R.match_grain(noisy, clean, edited, seed=0)
    b, _ = R.match_grain(noisy, clean, edited, seed=0)
    assert np.array_equal(a, b)
    assert np.array_equal(a[~edited], clean[~edited])
    added = (a - clean)[edited].std()
    assert 0.003 < added < 0.03, added           # roughly the photo's own noise level
    assert info["grain_sigma"]


def test_highlight_shoulder_is_monotonic_and_bounded():
    x = np.linspace(0, 5, 1000, dtype=np.float32)
    y = R.highlight_shoulder(x)
    assert np.all(np.diff(y) >= 0) and y.max() < 1.0
    assert np.array_equal(y[x <= 0.7], x[x <= 0.7])


def test_guided_alpha_stays_in_range():
    rng = np.random.default_rng(0)
    gray = rng.random((H, W)).astype(np.float32)
    a = R.guided_alpha(gray, xx > 160)
    assert a.min() >= 0 and a.max() <= 1
