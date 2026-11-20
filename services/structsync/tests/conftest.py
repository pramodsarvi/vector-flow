"""Shared fixtures. Nothing here (or in any test) may load a model: every model constructor is
replaced with one that raises, so an accidental model call fails loudly instead of downloading."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _no_model(*_a, **_k):
    raise AssertionError("tests must not load models")


@pytest.fixture(autouse=True)
def no_models(monkeypatch):
    import structsync.catalog as C
    import structsync.render as R
    import structsync.vision as V
    monkeypatch.setattr(V.Segmenter, "__init__", _no_model)
    monkeypatch.setattr(V.Geometry, "__init__", _no_model)
    monkeypatch.setattr(C.CatalogSearch, "__init__", _no_model)
    monkeypatch.setattr(R.Harmonizer, "__init__", _no_model)


# ---------------------------------------------------------------- synthetic geometry
W, H = 320, 240
K = np.array([[300.0, 0, W / 2], [0, 300.0, H / 2], [0, 0, 1]])
UP = np.array([0.0, -1.0, 0.0])          # camera coords: x right, y down, z forward


def rays():
    yy, xx = np.mgrid[0:H, 0:W]
    return np.stack([xx, yy, np.ones_like(xx)], -1).astype(np.float64) @ np.linalg.inv(K).T


def room_state(w=W, h=H):
    """Point map of a wall at z = 6 m above a floor 1.2 m below the camera, with a cabinet block on
    the wall: the arrays `Engine.restore` needs, with no models involved."""
    k = K.copy()
    k[0] *= w / W
    k[1] *= h / H
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.stack([xx, yy, np.ones_like(xx)], -1).astype(np.float64) @ np.linalg.inv(k).T
    t_wall = 6.0 / r[..., 2]          # far enough that the floor is visible (rows below ~3/4 height)
    t_floor = np.where(r[..., 1] > 1e-6, 1.2 / np.maximum(r[..., 1], 1e-6), np.inf)
    floor = t_floor < t_wall
    pts = r * np.where(floor, t_floor, t_wall)[..., None]
    cabinet = ~floor & (xx > 0.15 * w) & (xx < 0.45 * w) & (yy > 0.25 * h)
    wall = ~floor & ~cabinet
    return dict(points=pts.astype(np.float32), valid=np.ones((h, w), bool), K=k, up=UP.copy(),
                mask_floor=floor, mask_wall=wall, mask_cabinet=cabinet,
                mask_countertop=np.zeros((h, w), bool), mask_backsplash=np.zeros((h, w), bool),
                mask_occluder=np.zeros((h, w), bool))


def two_wall_room(w=W, h=H, pitch=0.0, yaw=0.0, counter=True, ceiling_h=1.3):
    """A room whose `wall` class is two different planes, with a counter 0.91 m above the floor.

    `room_state` has one wall and no counter, so it can exercise neither plane instancing nor the
    counter-height scale prior. Surfaces: floor 1.2 m below the camera, a counter band 0.91 m above
    it, a back wall at z = 6 m and a left wall at x = −1.5 m.

    `pitch` tilts the camera (radians), so gravity is no longer the camera's own −y and the
    pipeline has to *recover* "up" from the floor plane rather than keep its initial guess.
    `yaw` swings the back wall off the image plane, so the camera's x axis is neither parallel nor
    perpendicular to it — without that, a horizontal plane's default texture axis satisfies
    "aligned to the wall" by accident and `align_horizontal_axes` is untestable.
    Both default to 0, which is the plain axis-aligned room.

    `counter=False` drops the countertop, giving a room with no standard work surface — a bedroom
    or hallway, where the only height prior available is the ceiling. `ceiling_h` is the ceiling's
    height above the camera; the default puts it 2.50 m above the floor, exactly on the prior.
    """
    k = np.array([[0.9 * w, 0, w / 2], [0, 0.9 * w, h / 2], [0, 0, 1]])
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.stack([xx, yy, np.ones_like(xx)], -1).astype(np.float64) @ np.linalg.inv(k).T
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rx = np.array([[1.0, 0, 0], [0, cp, -sp], [0, sp, cp]])        # camera pitch
    ry = np.array([[cy, 0, sy], [0, 1.0, 0], [-sy, 0, cy]])        # back-wall swing
    down = rx @ np.array([0.0, 1.0, 0.0])                          # gravity in camera coords

    def t_plane(n, d):
        den = r @ np.asarray(n, float)
        return np.where(den > 1e-6, d / np.where(np.abs(den) < 1e-9, 1e-9, den), np.inf)

    band = (yy > 0.52 * h) & (yy < 0.66 * h)          # the counter is only visible in a band
    # the ceiling sits 1.3 m above the camera and the floor 1.2 m below it: exactly the 2.5 m
    # floor-to-ceiling the scale prior expects
    ts = np.stack([t_plane(down, 1.2),
                   np.where(band & counter, t_plane(down, 0.29), np.inf),
                   t_plane(rx @ ry @ np.array([0.0, 0, 1.0]), 6.0),
                   t_plane(rx @ np.array([-1.0, 0, 0]), 1.5),
                   t_plane(-down, ceiling_h)])
    which, t = ts.argmin(0), ts.min(0)
    ok = np.isfinite(t)
    pts = r * np.where(ok, t, 1.0)[..., None]
    return dict(points=pts.astype(np.float32), valid=ok, K=k, up=UP.copy(),
                mask_floor=(which == 0) & ok, mask_countertop=(which == 1) & ok,
                mask_wall=((which == 2) | (which == 3)) & ok,
                mask_ceiling=(which == 4) & ok,
                mask_cabinet=np.zeros((h, w), bool), mask_backsplash=np.zeros((h, w), bool),
                mask_occluder=np.zeros((h, w), bool))


def photo(w=400, h=300, seed=0):
    import cv2
    from PIL import Image
    rng = np.random.default_rng(seed)
    img = 0.5 + 0.3 * cv2.GaussianBlur(rng.random((h, w, 3)).astype(np.float32), (0, 0), 8)
    img += 0.01 * rng.standard_normal((h, w, 3)).astype(np.float32)        # sensor-like noise
    return Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
