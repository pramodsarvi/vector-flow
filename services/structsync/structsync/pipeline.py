"""End-to-end engine: analyse once per photo (cached Scene), then fast per-swap renders."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image

from .catalog import SKU, CatalogSearch, load_catalog
from .regions import (
    COMPONENT,
    DETECTION,
    PLANE,
    RID_SEP,
    SINGLE,
    by_kind,
    expand_assign,
    high_index,
    instancing,
    is_region,
    kind_of,
    make_rid,
    union_of_kind,
)
from .render import (
    DRIFT_LIMIT,
    EDGE_LOSS_LIMIT,
    Harmonizer,
    drift_deltaE,
    guided_alpha,
    match_grain,
    planar_reflections,
    project_materials,
    sku_recognition,
    verify,
)
from .surfaces import (
    Plane,
    Surface,
    align_horizontal_axes,
    compute_uv,
    drop_ragged,
    fit_surface,
    front_edge,
    orientation_filter,
    point_normals,
    scale_from_prior,
)
from .vision import Geometry, Segmenter, pick_device

log = logging.getLogger(__name__)

MAX_SIDE = 1024         # analysis resolution (models, masks, geometry, tap-to-correct)
RENDER_MAX_SIDE = 2048  # output resolution: textures are re-projected analytically at up to this size
MIN_REGION_PX = 200     # below this a region is detector crumbs, not a surface
# a class whose pixels total less than this share of the photo is a false positive rather than a
# small surface. Applied to the class as a whole, never to one region of it, so splitting a wall
# into three can't silently delete all three.
CLASS_NOISE_FRACTION = {"wall": 0.03, "floor": 0.03}


@dataclass
class Scene:
    image: Image.Image
    rgb: np.ndarray                 # float sRGB 0..1
    points: np.ndarray
    valid: np.ndarray
    K: np.ndarray
    up: np.ndarray
    masks: dict[str, np.ndarray]          # {region id: mask}, plus the non-region "occluder"
    surfaces: dict[str, Surface] = field(default_factory=dict)   # {region id: Surface}
    normals: np.ndarray | None = None
    notes: list[str] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)
    full_rgb: np.ndarray | None = None   # render-resolution photo (≥ analysis size)
    _hi: dict | None = None              # cached render-resolution surfaces; reset whenever surfaces change
    scale: float = 1.0                   # metric correction applied to the planes (counter-height prior)
    edges: dict = field(default_factory=dict)  # slab front faces, e.g. {"countertop#0": Surface}
    rid_high: dict = field(default_factory=dict)  # {kind: highest index ever minted}, never lowered


def _split_by_component(mask, min_px=MIN_REGION_PX):
    """Connected components of a class mask, largest first — one per physically separate run."""
    n, lab = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    parts = [(lab == i) for i in range(1, n)]
    parts = [p for p in parts if p.sum() >= min_px]
    return parts or [mask]


class Engine:
    EDGE_KINDS = ("countertop",)   # horizontal slabs whose visible front face gets the same material
    EDGE_THICKNESS_M = 0.03

    def __init__(self, edges=True, reflections=False, grain=True, material_maps=False, local_tint=False):
        self.edges = edges              # default for render(edges=None)
        self.grain = grain              # default for render(grain=None): match the photo's sensor noise
        self.material_maps = material_maps  # SKU normal/roughness maps (not yet validated on photos)
        self.local_tint = local_tint        # per-surface neutral-neighbour tint (not yet validated)
        self.reflections = reflections  # default for render(reflections=None); off: single-view guesswork
        self._harm_error = None
        self.device = pick_device()
        self._seg = self._geo = self._search = self._harm = None
        self.catalog = load_catalog()
        self.by_id = {s.id: s for s in self.catalog}

    # lazy model loading keeps UI start-up fast
    @property
    def seg(self):
        self._seg = self._seg or Segmenter(self.device)
        return self._seg

    @property
    def geo(self):
        self._geo = self._geo or Geometry(self.device)
        return self._geo

    @property
    def search(self) -> CatalogSearch:
        self._search = self._search or CatalogSearch(self.catalog, "cuda" if self.device == "cuda" else "cpu")
        return self._search

    @property
    def harm(self):
        if self._harm is None and self._harm_error is None:
            try:
                self._harm = Harmonizer(self.device)
            except Exception as e:  # noqa: BLE001  (missing weights, OOM on small GPUs/RAM, ...)
                self._harm_error = f"{type(e).__name__}: {str(e).splitlines()[0][:160]}"
                log.warning("harmoniser failed to load: %s", self._harm_error)
        return self._harm

    def preload(self, harmonizer=False):
        _ = self.seg, self.geo, self.search
        if harmonizer and self.harm is None:
            log.warning("harmoniser unavailable (%s); renders fall back to the projection", self._harm_error)

    # ------------------------------------------------------------ stage 1-3
    @staticmethod
    def prepare(img: Image.Image) -> tuple[Image.Image, Image.Image]:
        """(analysis image ≤ MAX_SIDE, render image ≤ RENDER_MAX_SIDE) from any input photo."""
        img = img.convert("RGB")
        r = RENDER_MAX_SIDE / max(img.size)
        full = img.resize((round(img.width * r), round(img.height * r)), Image.LANCZOS) if r < 1 else img
        s = MAX_SIDE / max(img.size)
        if s < 1:
            img = img.resize((round(img.width * s), round(img.height * s)), Image.LANCZOS)
        return img, full

    def scene_state(self, sc: Scene) -> dict[str, np.ndarray]:
        """Everything needed to rebuild a Scene without re-running the models (for persistence)."""
        st = dict(points=sc.points.astype(np.float32), valid=sc.valid, K=sc.K, up=sc.up)
        st.update({f"mask_{k}": v for k, v in sc.masks.items()})
        # region indices must stay monotonic across a restart, so the high-water mark travels too
        st.update({f"ridhigh_{k}": np.int32(v) for k, v in sc.rid_high.items()})
        return st

    def restore(self, img: Image.Image, st, notes=(), timings=None) -> Scene:
        """Rebuild a Scene from `scene_state` output + the original photo; surfaces are refitted, so
        tap-to-correct edits survive a restart."""
        img, full = self.prepare(img)
        masks = {}
        for k in st:
            if not k.startswith("mask_"):
                continue
            key = k[5:]
            # a project saved before regions existed stored one mask per class: it becomes that
            # class's region 0, unsplit, so its stored assignments keep resolving
            if is_region(key) and RID_SEP not in key:
                key = make_rid(key, 0)
            masks[key] = np.asarray(st[k], bool)
        # a scene stored before indices were monotonic has no mark; seed it from the masks present
        rid_high = {k[len("ridhigh_"):]: int(st[k]) for k in st if k.startswith("ridhigh_")}
        for key, m in masks.items():
            if is_region(key) and m.any():
                k = kind_of(key)
                rid_high[k] = max(rid_high.get(k, -1), high_index([key], k))
        sc = Scene(img, np.asarray(img, np.float32) / 255.0, np.asarray(st["points"], np.float32),
                   np.asarray(st["valid"], bool), np.asarray(st["K"], np.float64), np.asarray(st["up"], np.float64),
                   masks, notes=list(notes), timings=dict(timings or {}), rid_high=rid_high)
        sc.full_rgb = np.asarray(full, np.float32) / 255.0 if full.size != img.size else sc.rgb
        sc.normals = point_normals(sc.points, sc.valid)
        self.refit(sc)
        return sc

    def analyze(self, img: Image.Image) -> Scene:
        img, full = self.prepare(img)
        t = {}
        t0 = time.time()
        masks = self.seg.parse(img)
        t["segment"] = time.time() - t0
        t0 = time.time()
        g = self.geo.solve(img)
        t["geometry"] = time.time() - t0
        sc = Scene(img, np.asarray(img, np.float32) / 255.0, g["points"], g["valid"], g["K"],
                   np.array([0.0, -1.0, 0.0]), masks, timings=t)
        sc.full_rgb = np.asarray(full, np.float32) / 255.0 if full.size != img.size else sc.rgb
        f = sc.K[0, 0]
        sc.notes.append(f"focal ≈ {f:.0f}px (HFOV {np.degrees(2 * np.arctan(img.width / 2 / f)):.0f}°)")
        t0 = time.time()
        sc.normals = point_normals(sc.points, sc.valid)
        self._orient(sc)
        self._instance(sc)
        t["surfaces"] = time.time() - t0
        return sc

    # ------------------------------------------------------------ region identity
    def _instance(self, sc: Scene):
        """Divide each class into addressable regions — once, here, at analysis time.

        How a class divides depends on what the class is (see `regions.INSTANCING`): a wall splits
        by plane, so a fridge standing in front of it doesn't become two walls; counters and
        backsplashes split by connected component, so an island and the back counter can take
        different stone; cabinets arrive already split per door, because only the detector knows
        where one door ends. The floor never splits.

        Indices run largest region first, ties broken in raster order, so re-analysing the same
        photo produces the same ids and therefore the same cache key.
        """
        new = {}
        for kind in sorted({kind_of(k) for k in sc.masks if is_region(k)}):
            m = union_of_kind(sc.masks, kind)
            if m is None or not m.any():
                continue
            mode = instancing(kind)
            if mode == SINGLE:
                parts = [m]
            elif mode == PLANE:
                parts = self._split_by_plane(sc, kind, m)
            elif mode == DETECTION:
                parts = [v for v in by_kind(sc.masks, kind).values() if v.any()]
            elif mode == COMPONENT:
                parts = _split_by_component(m)
            else:
                parts = [m]
            parts = [p for p in parts if p.sum() >= MIN_REGION_PX] or [m]
            # whatever the split dropped (unfitted or invalid pixels) stays with the biggest part,
            # so a region's pixels are never silently lost from the scene
            parts.sort(key=lambda p: (-int(p.sum()), int(np.argmax(p))))
            left = m & ~np.logical_or.reduce(parts)
            if left.any():
                parts[0] = parts[0] | left
            for i, part in enumerate(parts):
                new[make_rid(kind, i)] = part
            sc.rid_high[kind] = len(parts) - 1
        for k, v in sc.masks.items():        # occluder and friends pass through untouched
            if not is_region(k):
                new[k] = v
        sc.masks, sc.surfaces = new, {}
        self.refit(sc)

    def _split_by_plane(self, sc, kind, mask):
        s = fit_surface(kind, mask, sc.points, sc.valid, sc.up)
        if len(s.planes) <= 1:
            return [mask]
        return [p.mask.copy() for p in s.planes]

    def refit(self, sc: Scene, keys=None):
        """Re-fit regions in place. Never splits or merges them: region ids are decided once, by
        `_instance`, so a material assignment and its cache key stay valid across corrections and
        restarts."""
        keys = list(keys) if keys is not None else [k for k in sc.masks if is_region(k)]
        sc._hi = None
        # gravity: refine "up" from the largest horizontal surface before fitting the rest. Asks
        # the class as a whole, since one floor region on its own may be too small to trust.
        for k in ("floor", "countertop"):
            m = union_of_kind(sc.masks, k)
            if m is not None and m.sum() > 2000:
                s = fit_surface(k, m, sc.points, sc.valid, np.array([0.0, -1.0, 0.0]))
                if s.planes and abs(s.planes[0].n[1]) > 0.6:  # tolerate camera pitch
                    n = s.planes[0].n
                    sc.up = n if n[1] < 0 else -n
                    break
        # walls/floors are large by nature: a class that totals almost nothing is detector noise
        noise = set()
        for kind, frac in CLASS_NOISE_FRACTION.items():
            m = union_of_kind(sc.masks, kind)
            if m is not None and m.sum() < frac * m.size:
                noise.add(kind)
        for k in keys:
            m = sc.masks.get(k)
            if m is None or m.sum() < MIN_REGION_PX or kind_of(k) in noise:
                sc.surfaces.pop(k, None)
                continue
            sc.surfaces[k] = fit_surface(k, m, sc.points, sc.valid, sc.up)
            sc.surfaces[k].rid = k
        # an emptied region (every pixel corrected away) is dropped rather than persisted forever
        for k in [k for k, m in sc.masks.items() if is_region(k) and not m.any()]:
            del sc.masks[k]
            sc.surfaces.pop(k, None)
        scale, note = scale_from_prior(sc.surfaces, sc.up)
        sc.notes = [n for n in sc.notes if not n.startswith(("scaled", "metric", "prior"))] + [note]
        align_horizontal_axes(sc.surfaces, sc.up)
        for surf in sc.surfaces.values():
            for p in surf.planes:
                p.d *= scale
            compute_uv(surf, sc.K, sc.rgb.shape[:2], sc.points, sc.valid)
        sc.scale = scale
        sc.edges = self._edges(sc, sc.surfaces, sc.K, sc.rgb.shape[:2], sc.points, sc.valid,
                               sc.masks.get("occluder"))

    def _edges(self, sc, surfaces, K, shape, points, valid, occluder):
        out = {}
        blocked = occluder if occluder is not None else np.zeros(shape, bool)
        for rid, s in by_kind(surfaces, *self.EDGE_KINDS).items():
            if s is not None and s.planes:
                e = front_edge(s, K, shape, points * sc.scale, valid, sc.up, blocked, self.EDGE_THICKNESS_M,
                               px_scale=shape[1] / sc.rgb.shape[1])
                if e is not None:
                    out[rid] = e
        return out

    def _orient(self, sc, keys=None):
        # first pass with camera "up", then again with the refined gravity
        for _ in range(2):
            for k in list(keys if keys is not None else [k for k in sc.masks if is_region(k)]):
                if k in sc.masks and sc.masks[k].any():
                    sc.masks[k] = orientation_filter(k, sc.masks[k], sc.normals, sc.up)
                    # auto-detections only; tap-to-correct additions are never dropped
                    if kind_of(k) == "cabinet":
                        sc.masks[k] = drop_ragged(sc.masks[k])
            self.refit(sc)

    def _mint_rid(self, sc: Scene, kind: str) -> str:
        """A fresh region id for `kind`. Monotonic per scene and never reused — see `high_index`.
        The scene's own high-water mark is used rather than the live masks, so a region that was
        corrected away does not free its index."""
        n = max(sc.rid_high.get(kind, -1), high_index(sc.masks, kind)) + 1
        sc.rid_high[kind] = n
        return make_rid(kind, n)

    def edit_mask(self, sc: Scene, kind: str, xy, add=True, region: str | None = None):
        """Tap-to-correct: a SAM 2 point prompt grows or shrinks a surface at the tapped pixel.

        `region` names the region to edit. Without one, a removal takes the tapped pixels away
        from every region of `kind`, and an addition *mints a new region* — that is the "nothing
        was detected here at all" case (kitchen4/5 auto-detect no cabinet), which has no region to
        grow yet. This is the only place outside `_instance` that creates a region id.
        """
        kind = kind_of(region) if region else kind
        m = self.seg.mask_from_point(sc.image, xy)
        if add:
            target = region or self._mint_rid(sc, kind)
            for k in list(sc.masks):      # the pixels belong to one region only
                if k != target:
                    sc.masks[k] = sc.masks[k] & ~m
            keep = orientation_filter(kind, m, sc.normals, sc.up, keep_largest=1)
            cur = sc.masks.get(target)
            sc.masks[target] = keep if cur is None else (cur | keep)
        else:
            for t in ([region] if region else list(by_kind(sc.masks, kind))):
                if t in sc.masks:
                    sc.masks[t] = sc.masks[t] & ~m
        self.refit(sc)

    # ------------------------------------------------------------ render resolution
    @staticmethod
    def _small(sc):
        H, W = sc.rgb.shape[:2]
        if sc.full_rgb is None or sc.full_rgb.shape[:2] == (H, W):
            return lambda x: x
        return lambda x: cv2.resize(np.ascontiguousarray(x, np.float32), (W, H), interpolation=cv2.INTER_AREA)

    def render_surfaces(self, sc: Scene) -> dict:
        """Surfaces at render resolution. Planes are exact, so UVs and footprints are recomputed
        analytically from rays at full size (no texture upsampling blur); masks are upsampled and
        snapped onto the full-resolution photo's edges. BOM areas stay on the analysis-res surfaces.
        Returns {surfaces, edges, K, points (metric-corrected), valid} at render resolution."""
        full = sc.full_rgb
        if full is None or full.shape[:2] == sc.rgb.shape[:2]:
            return dict(surfaces=sc.surfaces, edges=sc.edges, K=sc.K, points=sc.points * sc.scale,
                        valid=sc.valid)
        if sc._hi is not None:
            return sc._hi
        H, W = full.shape[:2]
        h, w = sc.rgb.shape[:2]
        sx, sy = W / w, H / h
        K = sc.K.copy()
        K[0, 0] *= sx
        K[1, 1] *= sy
        K[0, 2] = (K[0, 2] + 0.5) * sx - 0.5
        K[1, 2] = (K[1, 2] + 0.5) * sy - 0.5
        pts = cv2.resize(np.nan_to_num(sc.points).astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
        val = cv2.resize(sc.valid.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
        gray = cv2.cvtColor(full.astype(np.float32), cv2.COLOR_RGB2GRAY)

        def up(m):
            f = cv2.resize(m.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
            return guided_alpha(gray, f > 0.5, r=2, eps=1e-4) > 0.5

        hi = {}
        for k, s in sc.surfaces.items():
            m = up(s.mask)
            planes = [Plane(p.n, p.d, p.eu, p.ev, up(p.mask) & m, p.horizontal) for p in s.planes]
            hs = Surface(k, m, planes)
            compute_uv(hs, K, (H, W), pts, val)
            hs.area_m2 = s.area_m2
            hi[k] = hs
        occ = sc.masks.get("occluder")
        occ = up(occ) if occ is not None else None
        sc._hi = dict(surfaces=hi, edges=self._edges(sc, hi, K, (H, W), pts, val, occ), K=K,
                      points=pts * sc.scale, valid=val)
        return sc._hi

    # ------------------------------------------------------------ stage 4-7
    def render(self, sc: Scene, assign: dict[str, str | None], harmonize=False, strength=0.3,
               check_sku=True, edges=None, reflections=None, grain=None, material_maps=None,
               local_tint=None):
        """edges / reflections / grain / material_maps / local_tint: None → engine default. Each
        feature has its own switch so a bad result can be isolated by turning one off."""
        grain = self.grain if grain is None else grain
        material_maps = self.material_maps if material_maps is None else material_maps
        local_tint = self.local_tint if local_tint is None else local_tint
        edges = self.edges if edges is None else edges
        reflections = self.reflections if reflections is None else reflections
        t0 = time.time()
        # a caller may name a class ("wall") or one region ("wall#1"); a class covers all of its
        # regions, which is what an older client and the API examples both expect
        assign = expand_assign(assign, sc.surfaces.keys())
        skus: dict[str, SKU] = {k: self.by_id[v] for k, v in assign.items() if v in self.by_id}
        full = sc.full_rgb if sc.full_rgb is not None else sc.rgb
        geo = self.render_surfaces(sc)
        surfs, rskus = dict(geo["surfaces"]), dict(skus)
        if edges:  # slab front faces take the same SKU as their top; never part of the BOM
            for k, e in geo["edges"].items():
                if k in skus:
                    surfs[e.rid], rskus[e.rid] = e, skus[k]
        proj, edited, info = project_materials(full, surfs, rskus, up=sc.up, material_maps=material_maps,
                                               local_tint=local_tint)
        timings = {"projection": time.time() - t0}
        if reflections and edited.any():
            t0 = time.time()
            proj, _ = planar_reflections(proj, surfs, rskus, geo["K"], geo["points"], geo["valid"],
                                         px_scale=full.shape[1] / sc.rgb.shape[1])
            timings["reflections"] = time.time() - t0
        grain_info = {}
        if grain and edited.any():
            proj, grain_info = match_grain(full, proj, edited, seed=0)
        out = proj
        # structure checks use fixed pixel constants: always measure at analysis resolution
        small = self._small(sc)
        ed_small = np.zeros(sc.rgb.shape[:2], bool)
        for k in skus:
            sf = sc.surfaces.get(k)
            if sf is not None and sf.planes and sf.footprint is not None:
                ed_small |= sf.mask & (sf.footprint > 0)
            if edges and k in sc.edges:
                ed_small |= sc.edges[k].mask
        rep = verify(small(full), small(out), ed_small)
        rep.update(grain_info)
        rep["harmonized"] = False
        if harmonize and edited.any() and self.harm is None:
            rep["note"] = f"harmoniser unavailable ({self._harm_error}) → projection kept"
        elif harmonize and edited.any():
            t0 = time.time()
            prompt = "photo of a kitchen, " + \
                     ", ".join(f"{s.name} {kind_of(k)}" for k, s in skus.items()) + \
                     ", realistic lighting, high detail"
            try:
                # works at ≤1024 px internally; only its seam band is blended back into the full-res render
                h, band = self.harm(proj, edited, prompt, strength=strength, depth=sc.points[..., 2])
            except Exception as e:  # noqa: BLE001  (e.g. CUDA/host OOM mid-run)
                h = None
                rep["note"] = f"harmoniser failed ({type(e).__name__}) → projection kept"
            timings["harmonize"] = time.time() - t0
            if h is not None:
                drift = drift_deltaE(proj, h, info, band)
                rep["drift_deltaE"] = drift
                rep_h = verify(small(full), small(h), ed_small)
                edge_loss = rep["boundary_edge_recall"] - rep_h["boundary_edge_recall"]
                rep["harmonize_edge_loss"] = edge_loss
                # gates: projection is ground truth for the SKU; structural edges must survive
                if max(drift.values(), default=0) > DRIFT_LIMIT:
                    rep["note"] = (f"harmonised result rejected (seam ΔE drift {max(drift.values()):.1f} > "
                                   f"{DRIFT_LIMIT}) → projection kept")
                elif edge_loss > EDGE_LOSS_LIMIT:
                    rep["note"] = (f"harmonised result rejected (edge recall dropped {edge_loss:.2f} > "
                                   f"{EDGE_LOSS_LIMIT}) → projection kept")
                else:
                    out = h
                    rep.update(rep_h)
                    rep["harmonized"] = True
        if check_sku and skus:
            t0 = time.time()
            rep["sku_check"] = sku_recognition(self.search, out, info, skus)
            timings["sku_check"] = time.time() - t0
        # a bill of materials is ordered by material, not by region: regions sharing a SKU are one
        # line with their areas summed, and the line names the regions it covers
        agg: dict[tuple[str, str], dict] = {}
        for k, s in skus.items():
            surf = sc.surfaces.get(k)
            if not (surf and surf.planes):
                continue
            e = agg.setdefault((kind_of(k), s.id),
                               dict(surface=kind_of(k), sku=s.id, product=s.name, vendor=s.vendor,
                                    area=0.0, regions=[], price=float(s.price)))
            e["area"] += surf.area_m2
            e["regions"].append(k)
        bom = []
        for _, e in sorted(agg.items()):
            a, price = e.pop("area"), e.pop("price")
            bom.append(dict(**e, area_m2=round(a, 2), range=f"{a * 0.8:.1f}–{a * 1.2:.1f}",
                            est_cost=round(a * price)))
        rep["timings"] = timings
        return out, bom, rep
