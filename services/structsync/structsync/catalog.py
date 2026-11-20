"""Fixed catalogue: loading, demo-texture generation, and SigLIP 2 search.

Each SKU lives in catalog/<sku>/ with `albedo.png` (seamless tile) and the
entry in catalog/catalog.json gives physical tile size in metres.
Swap the demo SKUs for real supplier textures with the same schema.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

# override with STRUCTSYNC_CATALOG (e.g. a mounted volume in Docker)
ROOT = Path(os.environ.get("STRUCTSYNC_CATALOG", Path(__file__).resolve().parent.parent / "catalog"))


@dataclass
class SKU:
    id: str
    name: str
    category: str          # countertop | cabinet | backsplash | floor | wall | ceiling
    vendor: str
    price: float           # per m2
    tile_w: float          # metres covered by one texture tile (u direction)
    tile_h: float
    gloss: float           # 0..1, adds specular from the photo's highlights
    path: Path

    @property
    def albedo(self) -> np.ndarray:
        return np.asarray(Image.open(self.path / "albedo.png").convert("RGB"), np.float32) / 255.0

    @property
    def normal(self) -> np.ndarray | None:
        """Tangent-space normal map in [-1, 1] (OpenGL convention: +G = texture up), or None."""
        p = self.path / "normal.png"
        if not p.exists():
            return None
        n = np.asarray(Image.open(p).convert("RGB"), np.float32) / 127.5 - 1.0
        n[..., :2] = np.where(np.abs(n[..., :2]) < 0.006, 0.0, n[..., :2])   # 8-bit "flat" is 128, not 127.5
        return n

    @property
    def roughness(self) -> np.ndarray | None:
        """Roughness in [0, 1] (0 = mirror, 1 = matte), or None."""
        p = self.path / "roughness.png"
        if not p.exists():
            return None
        return np.asarray(Image.open(p).convert("L"), np.float32) / 255.0


def load_catalog() -> list[SKU]:
    if not (ROOT / "catalog.json").exists():
        build_demo_catalog()
    data = json.loads((ROOT / "catalog.json").read_text())
    return [SKU(path=ROOT / d["id"], **d) for d in data]


# ---------------------------------------------------------------- demo textures
# All generators are exactly periodic (seamless) in both axes: every noise field is built from a
# periodic lattice, and every sinusoid uses an integer number of periods per tile.
def _noise2(n, sx, sy, rng, octaves=5, gain=0.5):
    """Periodic fBm value noise with sx×sy lattice cells at the base octave (anisotropic if sx≠sy)."""
    out = np.zeros((n, n), np.float32)
    amp, tot = 1.0, 0.0
    for o in range(octaves):
        cx, cy = max(2, int(sx * 2 ** o)), max(2, int(sy * 2 ** o))
        g = np.tile(rng.random((cy, cx)).astype(np.float32), (3, 3))  # 3×3 tiling → wrap-safe interp
        up = cv2.resize(g, (3 * n, 3 * n), interpolation=cv2.INTER_CUBIC)[n:2 * n, n:2 * n]
        out += amp * up
        tot += amp
        amp *= gain
    return out / tot


def _noise(n, scale, rng, octaves=5):
    return _noise2(n, scale, scale, rng, octaves)


def _wrap_blur(img, sigma):
    """Gaussian blur that respects the tile wrap (no seam at the borders)."""
    p = int(3 * sigma) + 1
    big = np.pad(img, ((p, p), (p, p)) + ((0, 0),) * (img.ndim - 2), mode="wrap")
    return cv2.GaussianBlur(big, (0, 0), sigma)[p:-p, p:-p]


def _lerp(a, b, t):
    t = t[..., None]
    return np.array(a, np.float32) * (1 - t) + np.array(b, np.float32) * t


def _marble(n, base, vein, rng, fine=0.5):
    """Domain-warped veining: a few bold, branching veins plus fine secondary veins and cloudiness."""
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32) / n
    w1, w2 = _noise(n, 2, rng, 6), _noise(n, 3, rng, 6)
    # main veins: integer frequencies keep the phase periodic; the warp makes them meander
    ph = 2 * np.pi * (1 * xx + 1 * yy) + 3.2 * (w1 - 0.5) + 1.5 * (w2 - 0.5)
    width = 0.05 + 0.16 * _noise(n, 3, rng, 3) ** 2
    main = np.exp(-np.abs(np.sin(ph)) / width)
    # secondary hairline veins in another direction
    ph2 = 2 * np.pi * (2 * xx - 1 * yy) + 4.0 * (_noise(n, 4, rng, 6) - 0.5)
    sec = np.exp(-np.abs(np.sin(ph2)) / 0.025) * (_noise(n, 3, rng, 3) > 0.45)
    v = np.clip(main + fine * sec, 0, 1)
    v = _wrap_blur(v.astype(np.float32), 1.2) ** 1.3
    cloud = _noise(n, 4, rng, 6)
    base_col = _lerp(base, np.array(base) * 0.94 + np.array(vein) * 0.06, cloud)
    img = base_col * (1 - v[..., None]) + np.array(vein, np.float32) * v[..., None]
    return img * (0.97 + 0.03 * _noise(n, 64, rng, 3))[..., None]


def _granite(n, colors, weights, rng):
    """Crystalline granite: each mineral is a thresholded high-frequency noise layer, plus mica glints."""
    img = np.zeros((n, n, 3), np.float32) + np.array(colors[0], np.float32)
    img *= (0.92 + 0.16 * _noise(n, 24, rng, 3))[..., None]
    for c, w in zip(colors[1:], weights):
        f = _noise(n, 160, rng, 2)
        m = f > np.quantile(f, 1 - w)
        tone = (0.85 + 0.3 * _noise(n, 96, rng, 2))[..., None]
        img = np.where(m[..., None], np.array(c, np.float32) * tone, img)
    glint = _noise(n, 300, rng, 1) > 0.9
    img[glint] = np.clip(img[glint] * 1.35, 0, 1)
    return _wrap_blur(img, 0.6)


def _wood(n, light, dark, rng, planks=4, joints=True, figure=1.0, rings=(6, 12), maps=None):
    """Planked wood with grain running along x: per-plank tone and ring density, flat-sawn
    "cathedral" figure from a warped ring field, fine pores, and staggered butt joints.
    If `maps` is a dict, also fills maps["height"] / maps["roughness"] (seams and joints are
    recessed; open pores slightly lower and rougher). Uses no extra random draws, so the albedo
    is byte-identical with or without maps."""
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32) / n
    img = np.zeros((n, n, 3), np.float32)
    height = np.ones((n, n), np.float32)
    pores = _noise2(n, 4, 256, rng, 2)              # streaks along x
    ph = n // planks
    for i in range(planks):
        y0, y1 = i * ph, (i + 1) * ph if i < planks - 1 else n
        ys = slice(y0, y1)
        yl = (yy[ys] - y0 / n) * planks             # 0..1 across the plank
        warp = _noise2(n, 2, 8, rng, 4)[ys]
        rings_ = (rings[0] + (rings[1] - rings[0]) * rng.random()) * (
            yl + 0.35 * figure * np.sin(2 * np.pi * (xx[ys] + rng.random())) + 0.6 * figure * warp)
        r = rings_ - np.floor(rings_)
        late = np.clip((r - 0.65) / 0.25, 0, 1) * np.clip((1.0 - r) / 0.1, 0, 1)   # dark latewood band
        t = np.clip(0.25 * _noise2(n, 3, 24, rng, 3)[ys] + 0.55 * late + 0.25 * pores[ys], 0, 1)
        tone = 0.88 + 0.24 * rng.random()
        hue = np.array([1.0, 0.98 + 0.04 * rng.random(), 0.95 + 0.08 * rng.random()], np.float32)
        img[ys] = _lerp(light, dark, t) * tone * hue
        img[y0:y0 + max(1, n // 512) + 1] *= 0.6    # long-edge seam (small chamfer)
        height[y0:y0 + max(1, n // 512) + 1] = 0.0
        if joints:                                    # staggered end joint per plank
            x = int(rng.random() * n)
            img[ys, x:x + max(1, n // 512) + 1] *= 0.6
            height[ys, x:x + max(1, n // 512) + 1] = 0.0
    if maps is not None:
        maps["height"] = np.minimum(height, 1.0 - 0.08 * pores)
        maps["roughness"] = np.clip(0.55 + 0.25 * pores + 0.2 * (height < 0.5), 0, 1).astype(np.float32)
    return img


def _paint(n, color, rng):
    """Painted surface: very low-contrast roller mottling + fine grain."""
    mott = 0.985 + 0.03 * _noise(n, 6, rng, 4)
    grain = 0.995 + 0.01 * _noise(n, 128, rng, 2)
    return np.array(color, np.float32)[None, None] * (mott * grain)[..., None]


def _subway(n, tile, grout, rng, rows=4, cols=2, maps=None):
    """Running-bond glazed tile: per-tile tone, glaze undulation, cushion-edge darkening, recessed grout.
    If `maps` is a dict, also fills maps["height"] (cushioned tiles, recessed grout) and
    maps["roughness"] (glossy glaze, rough grout). No extra random draws."""
    img = np.zeros((n, n, 3), np.float32) + np.array(grout, np.float32)
    height = np.zeros((n, n), np.float32)
    rough = np.full((n, n), 0.85, np.float32)
    img *= (0.95 + 0.1 * _noise(n, 48, rng, 2))[..., None]
    th, tw, g = n // rows, n // cols, max(2, n // 160)
    yy, xx = np.mgrid[0:th - 2 * g, 0:tw - 2 * g].astype(np.float32)
    h, w = yy.shape
    edge = np.minimum.reduce([yy, xx, h - 1 - yy, w - 1 - xx])
    profile = np.clip(edge / (0.12 * h), 0, 1) ** 0.5   # 0 at the tile edge → 1 on the flat face
    cushion = 0.86 + 0.14 * profile
    glaze = _noise(n, 12, rng, 3)
    for r in range(rows):
        off = (tw // 2) * (r % 2)
        for c in range(-1, cols + 1):
            x0 = c * tw + off + g
            shade = (0.93 + 0.1 * rng.random()) * np.array([1, 1 + 0.02 * (rng.random() - .5), 1], np.float32)
            y0 = r * th + g
            for dx in range(w):                       # column copy keeps the wrap seamless
                X = (x0 + dx) % n
                img[y0:y0 + h, X] = np.array(tile, np.float32) * shade * cushion[:, dx, None] * \
                    (0.97 + 0.06 * glaze[y0:y0 + h, X, None])
                height[y0:y0 + h, X] = 0.4 + 0.6 * profile[:, dx]
                rough[y0:y0 + h, X] = 0.15 + 0.1 * glaze[y0:y0 + h, X]
    if maps is not None:
        maps["height"], maps["roughness"] = height, rough
    return img


def _transposed(img, maps):
    """Rotate a texture 90° (grain along v); keeps any height/roughness maps aligned with it."""
    if maps is not None:
        for k in list(maps):
            maps[k] = np.ascontiguousarray(maps[k].T)
    return np.transpose(img, (1, 0, 2))


def height_to_normal(height, max_tilt=0.6):
    """Tangent-space normal map from a periodic height field (wrap-aware Sobel), OpenGL convention
    (+G = texture up, i.e. towards row 0): n = (−∂h/∂col, +∂h/∂row, 1)/|·|. Slopes are scaled so
    the steepest gradient tilts the normal by `max_tilt` (tan of the angle). Returns [-1, 1] floats."""
    h = np.pad(height.astype(np.float32), 1, mode="wrap")
    gx = cv2.Sobel(h, cv2.CV_32F, 1, 0, ksize=3)[1:-1, 1:-1] / 8.0
    gy = cv2.Sobel(h, cv2.CV_32F, 0, 1, ksize=3)[1:-1, 1:-1] / 8.0
    k = max_tilt / max(float(np.percentile(np.hypot(gx, gy), 99.5)), 1e-6)
    n = np.stack([-gx * k, gy * k, np.ones_like(gx)], -1)
    return n / np.linalg.norm(n, axis=-1, keepdims=True)


def encode_normal(n):
    return np.clip(np.round((n + 1.0) * 127.5), 0, 255).astype(np.uint8)


DEMO = [
    # id, name, category, vendor, price, tile_w, tile_h, gloss, generator
    ("CT-CAL-01", "Calacatta White Quartz", "countertop", "StoneCo", 420, 1.6, 1.6, 0.6,
     lambda r: _marble(1024, (0.93, 0.92, 0.9), (0.55, 0.5, 0.45), r)),
    ("CT-NER-02", "Nero Marquina Marble", "countertop", "StoneCo", 510, 1.6, 1.6, 0.7,
     lambda r: _marble(1024, (0.08, 0.08, 0.09), (0.85, 0.85, 0.85), r)),
    ("CT-GRA-03", "Absolute Grey Granite", "countertop", "StoneCo", 290, 0.6, 0.6, 0.5,
     lambda r: _granite(1024, [(0.33, 0.33, 0.34), (0.12, 0.12, 0.13), (0.55, 0.55, 0.56), (0.05, 0.05, 0.06)],
                        [0.30, 0.18, 0.08], r)),
    ("CT-OAK-04", "Oak Butcher Block", "countertop", "TimberWorks", 260, 1.2, 1.2, 0.2,
     lambda r, m=None: _wood(1024, (0.80, 0.62, 0.42), (0.55, 0.37, 0.21), r, planks=8, figure=0.5, maps=m)),
    ("CB-NAV-01", "Navy Satin Paint", "cabinet", "PaintLab", 45, 0.5, 0.5, 0.3,
     lambda r: _paint(512, (0.12, 0.17, 0.3), r)),
    ("CB-SAG-02", "Sage Green Matte", "cabinet", "PaintLab", 45, 0.5, 0.5, 0.1,
     lambda r: _paint(512, (0.55, 0.62, 0.5), r)),
    ("CB-WHT-03", "Pure White Gloss", "cabinet", "PaintLab", 50, 0.5, 0.5, 0.6,
     lambda r: _paint(512, (0.93, 0.93, 0.91), r)),
    ("CB-WAL-04", "Walnut Veneer", "cabinet", "TimberWorks", 180, 0.8, 0.8, 0.25,
     lambda r, m=None: _transposed(_wood(1024, (0.45, 0.3, 0.2), (0.25, 0.15, 0.09), r, planks=1, joints=False,
                                         figure=0.25, rings=(28, 36), maps=m), m)),
    ("BS-SUB-01", "White Subway Tile 7.5x15", "backsplash", "TileHouse", 60, 0.3, 0.3, 0.7,
     lambda r, m=None: _subway(1024, (0.95, 0.95, 0.93), (0.75, 0.75, 0.73), r, maps=m)),
    ("BS-GRN-02", "Emerald Subway Tile", "backsplash", "TileHouse", 85, 0.3, 0.3, 0.8,
     lambda r, m=None: _subway(1024, (0.1, 0.4, 0.3), (0.8, 0.8, 0.78), r, maps=m)),
    ("FL-OAK-01", "Light Oak Plank Floor", "floor", "TimberWorks", 70, 1.0, 1.0, 0.2,
     lambda r, m=None: _wood(1024, (0.82, 0.68, 0.50), (0.62, 0.47, 0.32), r, planks=5, maps=m)),
    ("WL-GRG-01", "Greige Wall Paint", "wall", "PaintLab", 12, 0.5, 0.5, 0.05,
     lambda r: _paint(512, (0.72, 0.68, 0.62), r)),
    # Ceilings. Appended rather than grouped with the other paints on purpose: the demo albedos are
    # seeded by list index, so inserting anywhere above this point would regenerate every texture
    # below it and no longer match the committed catalogue.
    ("CL-WHT-01", "Flat White Ceiling", "ceiling", "PaintLab", 10, 2.0, 2.0, 0.02,
     lambda r: _paint(512, (0.95, 0.95, 0.94), r)),
    ("CL-OAK-02", "White Oak Plank Ceiling", "ceiling", "TimberWorks", 85, 1.2, 1.2, 0.15,
     lambda r, m=None: _wood(1024, (0.86, 0.76, 0.62), (0.68, 0.56, 0.42), r, planks=5, maps=m)),
    ("CL-ACO-03", "Acoustic Tile 600", "ceiling", "TileHouse", 35, 1.2, 1.2, 0.05,
     lambda r, m=None: _subway(1024, (0.90, 0.89, 0.86), (0.70, 0.69, 0.67), r, rows=2, cols=2, maps=m)),
]


def build_demo_catalog():
    ROOT.mkdir(parents=True, exist_ok=True)
    entries = []
    for i, (sid, name, cat, vendor, price, tw, th, gloss, gen) in enumerate(DEMO):
        rng = np.random.default_rng(i)
        img = np.clip(gen(rng), 0, 1)       # generators called without maps: unchanged albedo
        (ROOT / sid).mkdir(exist_ok=True)
        Image.fromarray((img * 255).astype(np.uint8)).save(ROOT / sid / "albedo.png")
        entries.append(dict(id=sid, name=name, category=cat, vendor=vendor, price=price,
                            tile_w=tw, tile_h=th, gloss=gloss))
    (ROOT / "catalog.json").write_text(json.dumps(entries, indent=2))


def write_demo_maps(root: Path | None = None, force=False) -> list[str]:
    """Add normal.png / roughness.png for demo SKUs that support them. Only *creates missing* maps
    and never touches albedo.png, catalog.json or the search embeddings. A SKU is only written if its
    albedo.png is byte-identical to the regenerated demo albedo, so a supplier texture that reuses a
    demo id never gets a mismatched relief map. Returns the SKU ids written.

        python -m structsync.catalog --write-demo-maps
    """
    import io
    root = Path(root or ROOT)
    done = []
    for i, (sid, *_rest, gen) in enumerate(DEMO):
        d = root / sid
        if not (d / "albedo.png").exists() or (not force and (d / "normal.png").exists()):
            continue
        maps = {}
        try:
            img = np.clip(gen(np.random.default_rng(i), maps), 0, 1)
        except TypeError:          # generator without map support (marble, granite, paint)
            continue
        if "height" not in maps:
            continue
        buf = io.BytesIO()
        Image.fromarray((img * 255).astype(np.uint8)).save(buf, "PNG")
        if buf.getvalue() != (d / "albedo.png").read_bytes():
            continue               # albedo was replaced: don't guess its relief
        Image.fromarray(encode_normal(height_to_normal(maps["height"]))).save(d / "normal.png")
        if "roughness" in maps and (force or not (d / "roughness.png").exists()):
            Image.fromarray((np.clip(maps["roughness"], 0, 1) * 255).astype(np.uint8)).save(d / "roughness.png")
        done.append(sid)
    return done


# ---------------------------------------------------------------- SigLIP 2 search
class CatalogSearch:
    """Text → SKU search. Image embeddings are computed once and cached to disk."""

    MODEL = "google/siglip2-base-patch16-224"

    def __init__(self, skus: list[SKU], device="cpu"):
        import torch
        from transformers import AutoModel, AutoProcessor
        self.torch, self.skus, self.device = torch, skus, device
        self.model = AutoModel.from_pretrained(self.MODEL).to(device).eval()
        self.proc = AutoProcessor.from_pretrained(self.MODEL)
        cache = ROOT / "siglip_embeddings.npy"
        if cache.exists() and len(np.load(cache)) == len(skus):
            self.emb = np.load(cache)
        else:
            imgs = [Image.open(s.path / "albedo.png").convert("RGB") for s in skus]
            with torch.no_grad():
                x = self.proc(images=imgs, return_tensors="pt").to(device)
                e = self.model.get_image_features(**x)
                e = getattr(e, "pooler_output", e)
            self.emb = (e / e.norm(dim=-1, keepdim=True)).cpu().numpy()
            np.save(cache, self.emb)

    def rank_image(self, img: Image.Image, category: str | None = None) -> list[tuple[SKU, float]]:
        torch = self.torch
        with torch.no_grad():
            x = self.proc(images=[img.convert("RGB")], return_tensors="pt").to(self.device)
            e = self.model.get_image_features(**x)
            e = getattr(e, "pooler_output", e)
        e = (e / e.norm(dim=-1, keepdim=True)).cpu().numpy()[0]
        scores = [(s, float(v @ e)) for s, v in zip(self.skus, self.emb) if not category or s.category == category]
        return sorted(scores, key=lambda p: -p[1])

    def search(self, text: str, category: str | None = None, k=3) -> list[tuple[SKU, float]]:
        torch = self.torch
        with torch.no_grad():
            x = self.proc(text=[f"a photo of {text}"], padding="max_length", max_length=64,
                          return_tensors="pt").to(self.device)
            t = self.model.get_text_features(**x)
            t = getattr(t, "pooler_output", t)
        t = (t / t.norm(dim=-1, keepdim=True)).cpu().numpy()[0]
        # blend image similarity with a cheap name match so names like "navy" always hit
        scores = []
        for s, e in zip(self.skus, self.emb):
            if category and s.category != category:
                continue
            name_hit = sum(w in s.name.lower() for w in text.lower().split()) * 0.1
            scores.append((s, float(e @ t) + name_hit))
        return sorted(scores, key=lambda p: -p[1])[:k]


if __name__ == "__main__":
    import sys
    if "--write-demo-maps" in sys.argv:
        print("wrote maps for:", write_demo_maps(force="--force" in sys.argv) or "nothing (all present or replaced)")
    else:
        print("usage: python -m structsync.catalog --write-demo-maps [--force]")
