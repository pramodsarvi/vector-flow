"""Stage 3: surface model — planes, metric UVs, approximate areas, scale prior."""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .regions import by_kind, kind_of, largest_of_kind

COUNTER_HEIGHT_M = 0.91  # domain prior: standard counter height above floor
CEILING_HEIGHT_M = 2.50  # domain prior: typical floor-to-ceiling height in a dwelling


@dataclass
class Plane:
    n: np.ndarray        # unit normal (camera coords, OpenCV: x right, y down, z fwd)
    d: float             # plane: n·X = d
    eu: np.ndarray       # in-plane texture axes (metres)
    ev: np.ndarray
    mask: np.ndarray     # pixels assigned to this plane
    horizontal: bool


@dataclass
class Surface:
    kind: str
    mask: np.ndarray
    planes: list[Plane] = field(default_factory=list)
    uv: np.ndarray | None = None        # (H,W,2) metres
    footprint: np.ndarray | None = None  # (H,W) metres per pixel (for mip selection)
    area_m2: float = 0.0
    rid: str = ""                        # region id ("wall#1"); defaults to the bare class name

    def __post_init__(self):
        self.rid = self.rid or self.kind


UP_FACING = ("countertop", "floor")      # a surface you look down onto
DOWN_FACING = ("ceiling",)               # a surface you look up at
HORIZONTAL_KINDS = UP_FACING + DOWN_FACING


def point_normals(points, valid):
    """Per-pixel normals from the MoGe point map (camera coords)."""
    P = cv2.GaussianBlur(np.where(valid[..., None], points, 0).astype(np.float32), (0, 0), 2)
    dx = np.gradient(P, axis=1)
    dy = np.gradient(P, axis=0)
    n = np.cross(dx, dy)
    n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-9
    return n


def orientation_filter(kind, mask, normals, up, keep_largest=8):
    """Geometry-aware cleanup: a horizontal kind must face the right way, a vertical one must not.
    Fixes detector boxes that bleed onto walls (e.g. 'countertop' masks covering the backsplash).
    `kind` may be a class name or a region id.

    The *sign* matters, not just the axis. Measured against `point_normals`, an up-facing surface
    (floor, countertop) comes out at n·up ≈ −1 and a down-facing one (ceiling) at ≈ +1 — the
    opposite way round from what the names suggest, because of the cross-product ordering. Testing
    |n·up| alone, as this did before ceilings existed, lets a ceiling pass as a floor and a floor
    pass as a ceiling: in one photo they are the two largest horizontal planes there are.
    """
    k = kind_of(kind)
    cos = normals @ up
    if k in UP_FACING:
        ok = cos < -0.8
    elif k in DOWN_FACING:
        ok = cos > 0.8
    else:
        ok = np.abs(cos) < 0.45
    m = (mask & ok).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1:
        return m > 0
    big = np.argsort(-stats[1:, cv2.CC_STAT_AREA])[:keep_largest] + 1
    min_area = 0.01 * stats[1:, cv2.CC_STAT_AREA].max()
    keep = [b for b in big if stats[b, cv2.CC_STAT_AREA] >= min_area]
    out = np.isin(lab, keep)
    return cv2.morphologyEx(out.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8)).astype(bool) & mask


def drop_ragged(mask, min_solidity=0.75, min_px=150):
    """Remove blotchy components (area / convex-hull area below `min_solidity`) and crumbs.
    Painted fronts are near-rectangular (measured ≥0.79 on the samples); SAM's partial masks on
    low-contrast white panels come out ragged (≤0.73) and render as speckled paint."""
    m = mask.astype(np.uint8)
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    keep = np.zeros_like(m)
    for c in cs:
        comp = np.zeros_like(m)
        cv2.drawContours(comp, [c], -1, 1, -1)
        comp &= m
        area = int(comp.sum())
        if area >= min_px and area / max(cv2.contourArea(cv2.convexHull(c)), 1.0) >= min_solidity:
            keep |= comp
    return keep.astype(bool)


def ransac_plane(P, iters=200, thr=0.015, rng=None):
    rng = rng or np.random.default_rng(0)
    best_inl = None
    for _ in range(iters):
        a, b, c = P[rng.choice(len(P), 3, replace=False)]
        n = np.cross(b - a, c - a)
        if np.linalg.norm(n) < 1e-9:
            continue
        n /= np.linalg.norm(n)
        d = n @ a
        # scale-adaptive threshold: depth noise grows with distance
        inl = np.abs(P @ n - d) < thr * np.maximum(1.0, np.linalg.norm(P, axis=1))
        if best_inl is None or inl.sum() > best_inl.sum():
            best_inl = inl
    # least-squares refine on inliers
    Q = P[best_inl]
    c = Q.mean(0)
    n = np.linalg.svd(Q - c)[2][-1]
    return n, float(n @ c), best_inl


def fit_surface(kind, mask, points, valid, up, max_planes=3):
    """`kind` may be a class name or a region id; the Surface records the class in `kind` and the
    region it was fitted for in `rid`."""
    s = Surface(kind_of(kind), mask.copy(), rid=kind)
    ys, xs = np.nonzero(mask & valid)
    if len(ys) < 200:
        return s
    P_all = points[ys, xs]
    remaining = np.ones(len(P_all), bool)
    rng = np.random.default_rng(0)
    planes = []
    for _ in range(max_planes):
        idx = np.nonzero(remaining)[0]
        if len(idx) < max(200, 0.12 * len(P_all)):
            break
        sub = idx if len(idx) < 6000 else rng.choice(idx, 6000, replace=False)
        n, d, _ = ransac_plane(P_all[sub], rng=rng)
        res = np.abs(P_all @ n - d) < 0.03 * np.maximum(1, np.linalg.norm(P_all, axis=1))
        new = res & remaining
        if new.sum() < 0.12 * len(P_all):
            break
        planes.append((n, d))
        remaining &= ~res
    # countertop / floor / ceiling: force a single horizontal plane (they're flat by definition)
    if kind_of(kind) in ("countertop", "floor", "ceiling") and planes:
        planes = [max(planes, key=lambda p: abs(p[0] @ up))]
    # assign every mask pixel to the plane with least residual
    res = np.stack([np.abs(P_all @ n - d) for n, d in planes], 1)
    owner = res.argmin(1)
    for i, (n, d) in enumerate(planes):
        if d < 0:  # orient normal toward camera: n·X = d with d > 0
            n, d = -n, -d
        horiz = abs(n @ up) > 0.85
        if horiz:
            eu = np.array([1.0, 0, 0]) - n * n[0]   # camera x projected onto plane
        else:
            g = -up - n * (n @ -up)                 # "down" in plane → texture v
            ev = g / np.linalg.norm(g)
            eu = np.cross(ev, n)
        eu /= np.linalg.norm(eu)
        ev = np.cross(n, eu) if horiz else ev
        m = np.zeros_like(mask)
        m[ys[owner == i], xs[owner == i]] = True
        m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)) > 0
        planes[i] = Plane(n, d, eu, ev, m & mask, horiz)
    s.planes = planes
    return s


def align_horizontal_axes(surfs: dict, up):
    """Horizontal planes default to the camera x axis for texture u, so stone veins, butcher-block
    staves and floor planks follow the camera instead of the room. Re-align u with the dominant
    vertical structure (backsplash > cabinet fronts > wall), i.e. run along the wall."""
    ref = None
    for k in ("backsplash", "cabinet", "wall"):
        for s in by_kind(surfs, k).values():
            if not s.planes:
                continue
            p = max(s.planes, key=lambda q: q.mask.sum())
            if not p.horizontal:
                ref = p.n
                break
        if ref is not None:
            break
    if ref is None:
        return
    for s in surfs.values():
        for p in s.planes:
            if not p.horizontal:
                continue
            eu = np.cross(p.n, ref)            # in the horizontal plane and parallel to the wall
            if np.linalg.norm(eu) < 0.2:
                continue
            eu /= np.linalg.norm(eu)
            if eu[0] < 0:                      # keep roughly left→right so u stays continuous
                eu = -eu
            p.eu, p.ev = eu, np.cross(p.n, eu)


def compute_uv(surface: Surface, K, shape, points=None, valid=None):
    H, W = shape
    uv = np.zeros((H, W, 2), np.float32)
    fp = np.zeros((H, W), np.float32)
    Kinv = np.linalg.inv(K)
    yy, xx = np.mgrid[0:H, 0:W]
    rays = np.stack([xx, yy, np.ones_like(xx)], -1).astype(np.float64) @ Kinv.T
    area = 0.0
    for p in surface.planes:
        m = p.mask
        r = rays[m]
        denom = r @ p.n
        t = p.d / np.where(np.abs(denom) < 1e-6, 1e-6, denom)
        X = r * t[:, None]  # exact ray–plane intersection: no depth noise in the texture
        # sanity: near-grazing rays / sign flips explode; fall back to the measured point
        if points is not None:
            P = points[m]
            dP = np.linalg.norm(P, axis=1)
            bad = (t <= 0) | (np.abs(np.linalg.norm(X, axis=1) - dP) > 0.35 * dP) | ~valid[m]
            Pp = P - ((P @ p.n) - p.d)[:, None] * p.n  # snap measured point onto plane
            X = np.where(bad[:, None], Pp, X)
        u, v = X @ p.eu, X @ p.ev
        uv[m] = np.stack([u, v], -1)
        # metres covered by one pixel ≈ distance / (f · |cos θ|)
        cos = np.abs(r @ p.n) / np.linalg.norm(r, axis=1)
        dist = np.linalg.norm(X, axis=1)
        f = K[0, 0]
        # isotropic mip uses one footprint for both axes: the grazing (1/cos) axis alone over-blurs
        # receding floors and counters, so use the geometric mean of the two axes
        foot = dist / (f * np.sqrt(np.maximum(cos, 0.05)))
        fp[m] = foot
        area += float(np.sum(np.minimum((dist / f) ** 2 / np.maximum(cos, 0.15), 0.01)))
    surface.uv, surface.footprint, surface.area_m2 = uv, fp, area
    return surface


# (upper kind, lower kind, real separation in metres, plausible measured range, weight)
# The weight is how tightly the real world fixes that distance. A kitchen counter is a standard
# 0.91 m almost everywhere; ceilings run anywhere from 2.3 to 3.0 m. So when a photo shows both
# pairs the counter dominates, and the ceiling pair only has to carry rooms that have no counter.
HEIGHT_PRIORS = (
    ("countertop", "floor", COUNTER_HEIGHT_M, (0.3, 2.0), 1.0),
    ("ceiling", "floor", CEILING_HEIGHT_M, (1.8, 4.5), 0.3),
)


def _facing_ok(kind, n, up) -> bool:
    """Does a plane fitted for `kind` face the way that class has to face?

    `Plane.horizontal` is unsigned (`|n·up| > 0.85`), so on its own it lets an up-facing plane
    stand in for a ceiling — a mezzanine floor, a tabletop the detector called ceiling, or a
    tap-to-correct slip. That is the same asymmetry `orientation_filter` had before ceilings
    existed, and this is its second consumer, so it needs the same signed test.
    """
    k = kind_of(kind)
    c = float(n @ up)
    if k in UP_FACING:
        return c < -0.85
    if k in DOWN_FACING:
        return c > 0.85
    return False            # only a horizontal surface has a height worth comparing


def _plane_gap(surfs, upper, lower, up):
    """How far the `upper` class's plane sits above the `lower` one's, in metres, or None.

    Signed, not absolute: a prior states that one surface is *above* another, so a pair measured
    the wrong way round is a mis-segmentation and must be refused rather than scaled by.
    """
    a, b = largest_of_kind(surfs, upper), largest_of_kind(surfs, lower)
    if not (a and b and a.planes and b.planes):
        return None
    pa, pb = a.planes[0], b.planes[0]
    if not (pa.horizontal and pb.horizontal):
        return None
    if not (_facing_ok(upper, pa.n, up) and _facing_ok(lower, pb.n, up)):
        return None
    # signed heights along up: a point on the plane is n*d
    gap = float((pa.n * pa.d) @ up - (pb.n * pb.d) @ up)
    return gap if gap > 0 else None


def scale_from_prior(surfs, up):
    """Metric correction from whatever standard real-world heights the photo happens to show.

    MoGe's depth is metric but drifts, and a known distance between two detected planes pins it.
    A kitchen offers the counter-to-floor pair. A bedroom, hallway or bathroom has no counter, so
    without the ceiling-to-floor pair those rooms fall back to raw depth and their areas drift by
    about ±20 % — which is the single biggest accuracy limit outside kitchens.
    """
    votes, rejected = [], []
    for upper, lower, expect, (lo, hi), weight in HEIGHT_PRIORS:
        h = _plane_gap(surfs, upper, lower, up)
        if h is None:
            continue
        if not lo <= h <= hi:
            rejected.append(f"{upper}/{lower} measured {h:.2f} m")
            continue
        votes.append((weight, expect / h, f"{upper} over {lower} {h:.2f}→{expect} m"))
    if not votes:
        why = "; ".join(rejected) if rejected else "no pair of known-height surfaces in view"
        return 1.0, ("prior rejected (" + why + ")" if rejected
                     else "metric depth only (" + why + ")")
    # scales multiply, so combine them in log space: a weighted geometric mean
    total = sum(w for w, _, _ in votes)
    s = float(np.exp(sum(w * np.log(r) for w, r, _ in votes) / total))
    s = float(np.clip(s, 0.7, 1.4))
    return s, f"scaled \u00d7{s:.2f} (" + ", ".join(d for _, _, d in votes) + ")"


def front_edge(surface: Surface, K, shape, points, valid, up, blocked, thickness=0.03, px_scale=1.0):
    """Visible front face of a horizontal slab (countertop edge), which no detector segments.

    From every near-side boundary pixel of a horizontal plane, drop `thickness` metres along gravity
    and project back to the image: the pixels in between are the slab's front face. A column is kept
    only if the measured point map agrees (a surface at roughly the same distance — not a wall
    behind) and no occluder is in the way. Returns a Surface with metric UVs that continue the top's
    texture around the edge, or None. Never counted in BOM areas. Pixel constants are tuned at analysis
    resolution (1024 px); `px_scale` = render width / analysis width keeps them equivalent at render size.
    """
    max_px = 60 * px_scale
    open_w = max(3, int(round(5 * px_scale)) | 1)
    min_w = 25 * px_scale
    H, W = shape
    Kinv = np.linalg.inv(K)
    band = np.zeros((H, W), bool)
    uv = np.zeros((H, W, 2), np.float32)
    fp = np.zeros((H, W), np.float32)
    for p in surface.planes:
        if not p.horizontal:
            continue
        m = p.mask
        nxt = np.zeros_like(m)
        nxt[:-2] = m[1:-1] | m[2:]
        ys, xs = np.nonzero(m & ~nxt)
        if len(ys) == 0:
            continue
        r = np.stack([xs, ys, np.ones_like(xs)], -1).astype(np.float64) @ Kinv.T
        X = r * (p.d / np.where(np.abs(r @ p.n) < 1e-6, 1e-6, r @ p.n))[:, None]
        X2 = X - up[None] * thickness
        q = X2 @ K.T
        L = q[:, 1] / q[:, 2] - ys
        ok = (q[:, 2] > 0) & (L >= 1.5) & (L <= max_px)
        # point-map agreement halfway down the face
        ym = np.clip(np.round(ys + L / 2).astype(int), 0, H - 1)
        P = points[ym, xs]
        dist = np.linalg.norm(X, axis=1)
        ok &= valid[ym, xs] & (np.abs(np.linalg.norm(P, axis=1) - dist) < 0.08 * dist)
        u0, v0 = X @ p.eu, X @ p.ev
        foot = thickness / np.maximum(L, 1)
        alive = ok.copy()
        for k in range(1, int(np.ceil(L[ok].max())) + 1 if ok.any() else 0):
            yk = ys + k
            sel = alive & (k <= np.ceil(L)) & (yk < H)
            yk_s, xs_s = yk[sel], xs[sel]
            hit = blocked[yk_s, xs_s] | m[yk_s, xs_s]
            idx = np.nonzero(sel)[0]
            alive[idx[hit]] = False                     # stop the column at the first occluder
            idx, yk_s, xs_s = idx[~hit], yk_s[~hit], xs_s[~hit]
            band[yk_s, xs_s] = True
            uv[yk_s, xs_s, 0] = u0[idx]
            uv[yk_s, xs_s, 1] = v0[idx] + thickness * k / L[idx]   # texture wraps over the edge
            fp[yk_s, xs_s] = foot[idx]
    # columns pass/fail individually: close small gaps, then drop crumbs
    band = cv2.morphologyEx(band.astype(np.uint8), cv2.MORPH_OPEN, np.ones((1, open_w), np.uint8)) > 0
    n, lab, st, _ = cv2.connectedComponentsWithStats(band.astype(np.uint8), 8)
    band = np.isin(lab, [i for i in range(1, n) if st[i, cv2.CC_STAT_WIDTH] >= min_w])
    band &= ~surface.mask & ~blocked
    if band.sum() < 50 * px_scale ** 2:
        return None
    s = Surface(surface.kind + "_edge", band, list(surface.planes[:1]),
                rid=surface.rid + "_edge")
    s.uv, s.footprint = uv, np.where(band, fp, 0).astype(np.float32)
    return s
