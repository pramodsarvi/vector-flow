"""Stage 5: deterministic texture projection + shading transfer.
Stage 6: optional low-strength SDXL inpaint harmonisation on seams.
Stage 7: verification metrics.
"""
from __future__ import annotations

import logging

import cv2
import numpy as np
from PIL import Image

from .catalog import SKU
from .regions import kind_of
from .surfaces import Surface

log = logging.getLogger(__name__)


def srgb_to_lin(x):
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def lin_to_srgb(x):
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1 / 2.4) - 0.055)


def luminance(lin):
    return lin @ np.array([0.2126, 0.7152, 0.0722], np.float32)


def mip_pyramid(tex, levels=7):
    pyr = [tex]
    for _ in range(levels - 1):
        if min(pyr[-1].shape[:2]) < 8:
            break
        pyr.append(cv2.pyrDown(pyr[-1]))
    return pyr


def sample_map(tex, tile_w, tile_h, surf: Surface):
    """Sample any texture map (H×W or H×W×C, already in the space it is blended in) at each pixel's
    metric UV with trilinear mip filtering (blend of the two nearest levels, so receding floors
    don't show mip bands)."""
    tex = tex.astype(np.float32)
    th, tw = tex.shape[:2]
    pyr = mip_pyramid(tex)
    m = surf.mask & (surf.footprint > 0)
    u, v = surf.uv[..., 0], surf.uv[..., 1]
    # texture pixels per screen pixel → mip level
    tpx = surf.footprint * tw / tile_w
    level = np.clip(np.log2(np.maximum(tpx, 1e-6)), 0, len(pyr) - 1)
    out = np.zeros(surf.mask.shape + tex.shape[2:], np.float32)
    for L, t in enumerate(pyr):
        wgt = np.clip(1.0 - np.abs(level - L), 0, 1)
        sel = m & (wgt > 0)
        if not sel.any():
            continue
        h, w = t.shape[:2]
        mx = np.mod((u / tile_w) * w, w).astype(np.float32)
        my = np.mod((v / tile_h) * h, h).astype(np.float32)
        s = cv2.remap(t, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        out[sel] += s[sel] * (wgt[sel, None] if tex.ndim == 3 else wgt[sel])
    return out


def sample_texture(sku: SKU, surf: Surface):
    """SKU albedo (linear) at each pixel's metric UV."""
    return sample_map(srgb_to_lin(sku.albedo), sku.tile_w, sku.tile_h, surf)


def relief_shading(sku: SKU, surf: Surface, up):
    """Per-pixel shading multiplier from the SKU's normal map, plus a specular weight from its
    roughness map. Either is None when the SKU has no such map.

    Normal map is tangent space (OpenGL: +G = texture up). Each pixel uses the basis of the plane
    that owns it: N = nx·eu + ny·(−ev) + nz·n_out with n_out = −plane.n (planes satisfy n·X = d > 0,
    so plane.n points away from the camera). Light = normalise(up + 0.5·n_out): overhead light that
    also faces the surface, so a perfectly flat normal map is an exact identity on every plane
    (walls, counters, floors). micro = (N·l)/(n_out·l), clipped to [0.6, 1.4] and applied at 80 %.
    Roughness 0.5 leaves the SKU's gloss unchanged; 0 doubles the highlight, 1 removes it.
    """
    nmap, rmap = sku.normal, sku.roughness
    micro = spec = None
    m = surf.mask & (surf.footprint > 0)
    if nmap is not None:
        N = sample_map(nmap, sku.tile_w, sku.tile_h, surf)
        N /= np.maximum(np.linalg.norm(N, axis=-1, keepdims=True), 1e-6)   # mips shorten normals
        micro = np.ones(m.shape, np.float32)
        up = np.asarray(up, np.float64)
        for p in surf.planes:
            pm = p.mask & m
            if not pm.any():
                continue
            n_out = -np.asarray(p.n, np.float64)
            l = up + 0.5 * n_out
            l /= np.linalg.norm(l)
            den = float(n_out @ l)
            if den < 0.25:                      # surface facing away from the assumed light: skip
                continue
            Np = N[pm].astype(np.float64)
            W = Np[:, :1] * p.eu + Np[:, 1:2] * (-np.asarray(p.ev)) + Np[:, 2:3] * n_out
            W /= np.maximum(np.linalg.norm(W, axis=1, keepdims=True), 1e-9)
            r = np.clip((W @ l) / den, 0.6, 1.4)
            micro[pm] = (1.0 + 0.8 * (r - 1.0)).astype(np.float32)
    if rmap is not None:
        R = sample_map(rmap, sku.tile_w, sku.tile_h, surf)
        spec = np.where(m, np.clip(2.0 * (1.0 - R), 0, 2), 1.0).astype(np.float32)
    return micro, spec


def neighbour_tint(img_lin, m, exclude, tint, cap=0.06, min_px=400):
    """Local illuminant colour from untouched, near-neutral pixels around a surface (e.g. under a warm
    pendant vs. next to a cool window). Only low-saturation pixels count, so a red wall or a teal
    island next to the surface can't tint it; the shift from the global tint is capped at ±`cap`."""
    k = max(9, int(0.025 * np.hypot(*m.shape)) | 1)
    ring = cv2.dilate(m.astype(np.uint8), np.ones((k, k), np.uint8)).astype(bool) & ~m & ~exclude
    if ring.sum() < min_px:
        return tint
    px = img_lin[ring]
    mx, mn = px.max(1), px.min(1)
    lum = px @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    # saturation < 0.4: a white wall under warm/cool light (R/B ≈ 1.25/0.8 → 0.36) still counts; strongly
    # coloured neighbours (red wall, teal island ≈ 0.8–0.9) don't. The ±cap bounds what beige walls can do.
    ok = ((mx - mn) / np.maximum(mx, 1e-4) < 0.4) & (lum > 0.03) & (lum < 0.9)
    if ok.sum() < min_px:
        return tint
    local = px[ok].mean(0)
    local = (local / local.mean()) ** 0.5
    return (tint * np.clip(local / tint, 1 - cap, 1 + cap)).astype(np.float32)


def highlight_shoulder(x, knee=0.7):
    """Filmic roll-off above `knee` (linear): bright photos push white stone past 1.0, and a hard clip
    flattens its veins into a featureless white patch. Asymptotically approaches 1."""
    over = np.maximum(x - knee, 0)
    return np.where(x > knee, knee + (1 - knee) * (1 - np.exp(-over / (1 - knee))), x).astype(np.float32)


def _masked_blur(x, mf, sig):
    """Gaussian blur that only averages pixels inside the mask (no light bleeding in from neighbours)."""
    num = cv2.GaussianBlur(x * mf, (0, 0), sig)
    den = cv2.GaussianBlur(mf, (0, 0), sig)
    return num / (den + 1e-6)


def guided_alpha(guide_gray, mask, r=4, eps=2e-3):
    """Edge-aware soft alpha: guided filter (He et al.) of the binary mask, steered by the photo.
    Snaps jagged segmentation/normal-filter edges onto the real image edges."""
    I = guide_gray.astype(np.float32)
    p = mask.astype(np.float32)
    k = (2 * r + 1, 2 * r + 1)
    box = lambda x: cv2.boxFilter(x, -1, k, borderType=cv2.BORDER_REFLECT)  # noqa: E731
    mI, mp = box(I), box(p)
    a = (box(I * p) - mI * mp) / (box(I * I) - mI * mI + eps)
    b = mp - a * mI
    q = box(a) * I + box(b)
    return np.clip(q, 0, 1)


def shading_layer(img_lin, mask):
    """Illumination of the original surface, decomposed into
    - S:      large-scale light falloff / cast shadows (normalised to the surface median)
    - D:      fine *darkening* detail (door grooves, panel bevels, handle and contact shadows),
              attenuated when the old surface is itself patterned so its old pattern doesn't leak
    - hi:     strong specular highlights (lamp / window reflections) for glossy materials
    - exposure: scene-level white point
    """
    L = luminance(img_lin).astype(np.float32)
    diag = np.hypot(*L.shape)
    mf = mask.astype(np.float32)
    inside = L[mask] if mask.any() else L.ravel()
    # clamp highlights before low-passing so reflections don't inflate the shading estimate
    p97 = float(np.percentile(inside, 97))
    Lc = np.minimum(L, p97)
    low = _masked_blur(Lc, mf, max(2.0, diag * 0.006))
    med = float(np.median(inside))
    # dark originals (black granite, navy paint) have a tiny median: reflections and noise get
    # amplified when divided by it. Floor the denominator and work in log space with a tighter range.
    ref = max(med, 0.25 * float(np.percentile(inside, 75)), 1e-3)
    S = np.exp(np.clip(np.log(np.maximum(low, 1e-5) / ref), np.log(0.2), np.log(1.3))).astype(np.float32)

    # fine detail: ratio to a small-radius masked blur, keep only darkening (grooves/shadows darken;
    # bright old patterns such as white veins on dark stone are dropped)
    fine = _masked_blur(L, mf, max(1.5, diag * 0.002))
    ratio = np.clip(L / np.maximum(fine, 1e-5), 0, 1.5)
    D = np.minimum(ratio, 1.0)
    # how patterned was the original? uniform paint → keep detail; marble/granite/tile → suppress
    logr = np.log(np.maximum(ratio[mask], 1e-3)) if mask.any() else np.zeros(1)
    texture = float(np.median(np.abs(logr)))  # robust: sparse grooves barely move the median
    keep = float(np.clip(1.0 - (texture - 0.015) / 0.05, 0.15, 1.0))
    D = (1.0 - keep * (1.0 - D)).astype(np.float32)

    # absolute exposure from the whole scene: bright non-clipped pixels ≈ white-ish
    # surfaces (albedo ~0.75). Using the old surface would make dark→light swaps grey.
    Lg = L[L < 0.95]
    exposure = float(np.clip(np.percentile(Lg, 90) / 0.75, 0.2, 1.5)) if Lg.size else 1.0
    # specular: only the strongest, locally-peaked highlights (not every bright pattern pixel)
    thr = np.maximum(low * 1.6, p97)
    hi = np.clip(L - thr, 0, None)
    hi = cv2.GaussianBlur(hi, (0, 0), 1.0)
    return S, D, exposure, hi, dict(texture=texture, detail_keep=keep)


def project_materials(image: np.ndarray, surfaces: dict[str, Surface], assign: dict[str, SKU], up=None,
                      material_maps=False, local_tint=False):
    """image: float RGB sRGB 0..1. Returns (render sRGB, edited mask, per-surface info).
    material_maps: use SKU normal/roughness maps when present (needs `up`).
    local_tint: per-surface illuminant colour from neutral neighbouring pixels (capped)."""
    img_lin = srgb_to_lin(image).astype(np.float32)
    # global illuminant tint (gray world, softened) so new materials pick up the room colour
    tint = img_lin.reshape(-1, 3).mean(0)
    tint = (tint / tint.mean()) ** 0.5
    out = img_lin.copy()
    edited = np.zeros(image.shape[:2], bool)
    gray = cv2.cvtColor(np.clip(image, 0, 1).astype(np.float32), cv2.COLOR_RGB2GRAY)
    info = {}
    if local_tint:   # every surface being re-textured is excluded from every neighbour ring
        all_m = np.zeros(image.shape[:2], bool)
        for kind, sku in assign.items():
            sf = surfaces.get(kind)
            if sku is not None and sf is not None and sf.footprint is not None:
                all_m |= sf.mask
    for kind, sku in assign.items():
        surf = surfaces.get(kind)
        if sku is None or surf is None or not surf.planes:
            continue
        m = surf.mask & (surf.footprint > 0)
        if not m.any():
            continue
        alb = sample_texture(sku, surf)
        S, D, exp, hi, stats = shading_layer(img_lin, m)
        # pixels at the soft edge that lie just outside the mask have no UV: borrow the nearest texel
        ring = cv2.dilate(m.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool) & ~m
        if ring.any():
            _, lab = cv2.distanceTransformWithLabels((~m).astype(np.uint8), cv2.DIST_L2, 5,
                                                     labelType=cv2.DIST_LABEL_PIXEL)
            ys, xs = np.nonzero(m)
            src = np.zeros((lab.max() + 1, 3), np.float32)
            src[lab[ys, xs]] = alb[ys, xs]
            alb[ring] = src[lab[ring]]
        if kind.endswith("_edge") and kind[:-5] in surfaces:
            # vertical slab face: keep the original face-vs-top brightness ratio (faces under ceiling
            # light are darker than tops), clipped so a black old edge doesn't turn the new one black
            Lo = luminance(img_lin)
            top = surfaces[kind[:-5]].mask
            near = cv2.dilate(m.astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool) & top
            if near.sum() > 20:
                ratio = float(np.median(Lo[m]) / max(np.median(Lo[near]), 1e-4))
                ts = shading_layer(img_lin, top)
                S = np.full_like(S, float(np.median(ts[0][near])) * np.clip(ratio, 0.5, 0.9))
                exp = ts[2]
            hi = np.zeros_like(hi)
        t_k = neighbour_tint(img_lin, m, all_m, tint) if local_tint else tint
        micro = spec = None
        if material_maps and up is not None and not kind.endswith("_edge"):
            micro, spec = relief_shading(sku, surf, up)
        shade = S * D * exp if micro is None else S * D * exp * micro
        gloss = sku.gloss * hi if spec is None else sku.gloss * hi * spec
        new = alb * shade[..., None] * t_k + gloss[..., None]
        new = highlight_shoulder(new)
        # edge-aware alpha: follows real image edges, ~1 px soft, never grows more than 2 px
        # the guided filter snaps the boundary onto real image edges (±1 px); the composite itself
        # stays hard, because a soft blend weakens the structural edges (measured: edge recall
        # 0.995 hard vs 0.980 soft) and lets the old surface colour bleed back in as a halo
        g = guided_alpha(gray, m, r=2, eps=1e-4)
        inner = cv2.erode(m.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        outer = cv2.dilate(m.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        a = (inner | (outer & (g >= 0.5))).astype(np.float32)[..., None]
        out = out * (1 - a) + new * a
        edited |= m
        info[kind] = dict(S=S, exp=exp, tint=t_k, mask=m, relief=micro is not None, **stats)
    return lin_to_srgb(out).astype(np.float32), edited, info


# ---------------------------------------------------------------- camera grain matching
def _unit_grain(shape, seed, corr=0.6):
    """Spatially-correlated (demosaic-like) noise, luma-dominant like JPEG'd sensor noise; unit std."""
    rng = np.random.default_rng(seed)
    luma = cv2.GaussianBlur(rng.standard_normal(shape[:2]).astype(np.float32), (0, 0), corr)
    chroma = cv2.GaussianBlur(rng.standard_normal(shape).astype(np.float32), (0, 0), corr * 1.5)
    n = 0.85 * luma[..., None] + 0.5 * chroma
    return n / (n.std() + 1e-8)


def _residual(x, sigma=1.0):
    return x - cv2.GaussianBlur(x, (0, 0), sigma)


def match_grain(original, render, edited, bins=6, seed=0):
    """Give re-textured pixels the photo's own sensor/JPEG noise. Clean synthetic textures next to
    noisy real pixels are a strong "pasted on" cue.

    Noise level is measured as a function of brightness (shot noise grows with signal) from flat,
    unclipped areas of the original; the render's existing high-frequency content in the same
    brightness band is subtracted (shading detail already carries some real noise), and only the
    missing variance is added. Edited pixels only, so outside_change stays 0. Deterministic.
    """
    if not edited.any():
        return render, {}
    I = np.ascontiguousarray(original, np.float32)
    R = np.ascontiguousarray(render, np.float32)
    gray = cv2.cvtColor(I, cv2.COLOR_RGB2GRAY)
    gb = cv2.GaussianBlur(gray, (0, 0), 2.0)
    grad = np.hypot(cv2.Sobel(gb, cv2.CV_32F, 1, 0), cv2.Sobel(gb, cv2.CV_32F, 0, 1))
    Lo = cv2.GaussianBlur(gray, (0, 0), 1.0)
    ok = (Lo > 0.03) & (Lo < 0.95)
    flat_o = ok & ~edited & (grad < np.percentile(grad[ok], 35))
    if flat_o.sum() < 2000:
        return render, {}
    res_o = _residual(I)
    Rg = cv2.cvtColor(R, cv2.COLOR_RGB2GRAY)
    Lr = cv2.GaussianBlur(Rg, (0, 0), 1.0)
    gr = cv2.GaussianBlur(Rg, (0, 0), 2.0)
    grad_r = np.hypot(cv2.Sobel(gr, cv2.CV_32F, 1, 0), cv2.Sobel(gr, cv2.CV_32F, 0, 1))
    flat_r = edited & (grad_r < np.percentile(grad_r[edited], 35))
    res_r = _residual(R)
    edges_ = np.quantile(Lo[flat_o], np.linspace(0, 1, bins + 1))
    centers, s_add = [], []
    mad = lambda x: 1.4826 * np.median(np.abs(x - np.median(x, 0)), 0)  # noqa: E731
    for i in range(bins):
        sel_o = flat_o & (Lo >= edges_[i]) & (Lo <= edges_[i + 1])
        sel_r = flat_r & (Lr >= edges_[i]) & (Lr <= edges_[i + 1])
        if sel_o.sum() < 200:
            continue
        so = mad(res_o[sel_o])
        sr = mad(res_r[sel_r]) if sel_r.sum() > 200 else np.zeros(3, np.float32)
        centers.append(0.5 * (edges_[i] + edges_[i + 1]))
        s_add.append(np.sqrt(np.maximum(so ** 2 - sr ** 2, 0)))
    if not centers:
        return render, {}
    centers, s_add = np.array(centers), np.array(s_add)
    n = _unit_grain(R.shape, seed)
    k = float(np.std(_residual(n)))           # how much of unit grain the residual operator sees
    sig = np.stack([np.interp(Lr, centers, s_add[:, c]) for c in range(3)], -1) / max(k, 1e-6)
    out = np.where(edited[..., None], np.clip(R + n * sig, 0, 1), R).astype(np.float32)
    return out, dict(grain_sigma=[round(float(x), 4) for x in s_add.mean(0)])


# ---------------------------------------------------------------- planar reflections (optional)
REFLECT_MIN_GLOSS = 0.5   # only polished SKUs (quartz, marble, gloss tile) get mirror reflections


def planar_reflections(render, surfaces, skus, K, points, valid, steps=40, max_dist=3.0, px_scale=1.0):
    """Screen-space mirror reflections on glossy horizontal planes, applied to the *composited* render
    so recoloured cabinets are what gets reflected.

    For each pixel the view ray is mirrored about the plane and marched through the measured point
    map; the first sample that passes behind a measured surface (within a depth tolerance) is the
    reflected point. Rays that leave the image or find no agreeing surface add nothing — a single
    view can't see what's behind the camera, so misses are left as the diffuse render, never smeared.
    Strength = gloss × Schlick Fresnel (F0 = 0.04), so reflections only show at grazing angles.
    Returns (render sRGB, reflection-weight map).
    """
    H, W = render.shape[:2]
    lin = srgb_to_lin(render).astype(np.float32)
    Kinv = np.linalg.inv(K)
    out = lin.copy()
    weight = np.zeros((H, W), np.float32)
    dists = np.geomspace(0.02, max_dist, steps)
    depth = np.linalg.norm(points, axis=-1)
    for kind, sku in skus.items():
        surf = surfaces.get(kind)
        if surf is None or sku.gloss < REFLECT_MIN_GLOSS or surf.footprint is None:
            continue
        for p in surf.planes:
            if not p.horizontal:
                continue
            m = p.mask & (surf.footprint > 0)
            ys, xs = np.nonzero(m)
            if len(ys) == 0:
                continue
            r = np.stack([xs, ys, np.ones_like(xs)], -1).astype(np.float64) @ Kinv.T
            X = r * (p.d / np.where(np.abs(r @ p.n) < 1e-6, 1e-6, r @ p.n))[:, None]
            v = X / np.linalg.norm(X, axis=1, keepdims=True)
            rr = v - 2 * (v @ p.n)[:, None] * p.n[None]
            cos = np.abs(v @ p.n)
            hit_col = np.zeros((len(ys), 3), np.float32)
            hit = np.zeros(len(ys), bool)
            alive = np.ones(len(ys), bool)
            for dist in dists:
                idx = np.nonzero(alive & ~hit)[0]
                if len(idx) == 0:
                    break
                Y = X[idx] + rr[idx] * dist
                q = Y @ K.T
                inb = q[:, 2] > 1e-3
                u = np.where(inb, q[:, 0] / np.maximum(q[:, 2], 1e-3), -1)
                w = np.where(inb, q[:, 1] / np.maximum(q[:, 2], 1e-3), -1)
                inside = inb & (u >= 0) & (u < W - 1) & (w >= 0) & (w < H - 1)
                alive[idx[~inside]] = False             # left the image: no information, add nothing
                idx, u, w, Y = idx[inside], u[inside], w[inside], Y[inside]
                ui, wi = u.astype(int), w.astype(int)
                dY = np.linalg.norm(Y, axis=1)
                dP = depth[wi, ui]
                ok = valid[wi, ui] & ~m[wi, ui] & (dP <= dY) & (dY - dP < 0.08 * dY + 0.03)
                hit[idx[ok]] = True
                hit_col[idx[ok]] = lin[wi[ok], ui[ok]]
            F = 0.04 + 0.96 * (1 - cos) ** 5
            wgt = np.zeros((H, W), np.float32)
            col = np.zeros((H, W, 3), np.float32)
            wgt[ys[hit], xs[hit]] = (sku.gloss * F[hit]).astype(np.float32)
            col[ys[hit], xs[hit]] = hit_col[hit]
            # polished but not a perfect mirror: blur by roughness, normalised by hit coverage
            sig = (1.0 + 4.0 * (1 - sku.gloss)) * px_scale
            cov = cv2.GaussianBlur((wgt > 0).astype(np.float32), (0, 0), sig)
            col = cv2.GaussianBlur(col, (0, 0), sig) / np.maximum(cov, 1e-4)[..., None]
            wgt = cv2.GaussianBlur(wgt, (0, 0), sig) * m
            out = out * (1 - wgt[..., None]) + col * wgt[..., None]
            weight = np.maximum(weight, wgt)
    return lin_to_srgb(out).astype(np.float32), weight


# ---------------------------------------------------------------- harmonisation
class Harmonizer:
    """SDXL-Inpainting (OpenRAIL++) + depth ControlNet (OpenRAIL++) at low strength, seams only.

    Depth conditioning comes from the MoGe point map, so the generative pass is pinned to the
    measured geometry. Falls back to plain inpainting if the ControlNet can't be loaded.
    """

    MODEL = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
    VAE = "madebyollin/sdxl-vae-fp16-fix"
    CONTROLNET = "diffusers/controlnet-depth-sdxl-1.0"

    def __init__(self, device, use_controlnet=True):
        import torch
        from diffusers import AutoencoderKL
        self.torch = torch
        dtype = torch.float16 if device != "cpu" else torch.float32
        variant = "fp16" if dtype == torch.float16 else None
        import os
        # never start a multi-GB download implicitly from a render request: weights must be fetched with
        # `setup_gpu.sh --with-harmonizer` (or set STRUCTSYNC_ALLOW_DOWNLOAD=1)
        local = dict(local_files_only=os.environ.get("STRUCTSYNC_ALLOW_DOWNLOAD") != "1")
        vae = AutoencoderKL.from_pretrained(self.VAE, torch_dtype=dtype, **local)
        self.controlnet = False
        if use_controlnet:
            try:
                from diffusers import ControlNetModel, StableDiffusionXLControlNetInpaintPipeline
                cn = ControlNetModel.from_pretrained(self.CONTROLNET, torch_dtype=dtype, variant=variant, **local)
                self.pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
                    self.MODEL, controlnet=cn, vae=vae, torch_dtype=dtype, variant=variant, **local)
                self.controlnet = True
            except Exception as e:  # noqa: BLE001
                log.warning("ControlNet unavailable (%s); harmoniser uses plain inpainting", e)
        if not self.controlnet:
            from diffusers import AutoPipelineForInpainting
            self.pipe = AutoPipelineForInpainting.from_pretrained(
                self.MODEL, vae=vae, torch_dtype=dtype, variant=variant, **local)
        # SDXL + ControlNet (~9 GB fp16) doesn't fit next to the analysis models on ≤12 GB cards:
        # stream sub-models to the GPU one at a time instead
        vram = torch.cuda.get_device_properties(0).total_memory if device == "cuda" else 0
        if device == "cuda" and vram < 14 * 2 ** 30:
            self.pipe.enable_model_cpu_offload()
            self.pipe.vae.enable_tiling()
        else:
            self.pipe = self.pipe.to(device)
        self.pipe.set_progress_bar_config(disable=True)

    def __call__(self, render: np.ndarray, edited: np.ndarray, prompt: str, strength=0.3, steps=20,
                 depth: np.ndarray | None = None, seed=0):
        H, W = edited.shape
        # seam band = boundary of edited region
        k = max(3, int(0.012 * max(H, W)))
        e = edited.astype(np.uint8)
        band = (cv2.dilate(e, np.ones((k, k))) - cv2.erode(e, np.ones((k, k)))) > 0
        if not band.any():
            return render, band
        s = 1024 / max(H, W)
        w8, h8 = int(W * s) // 8 * 8, int(H * s) // 8 * 8
        img = Image.fromarray((np.clip(render, 0, 1) * 255).astype(np.uint8)).resize((w8, h8), Image.LANCZOS)
        msk = Image.fromarray((band * 255).astype(np.uint8)).resize((w8, h8), Image.NEAREST)
        kw = dict(prompt=prompt, negative_prompt="cartoon, blurry, distorted, warped lines, text",
                  image=img, mask_image=msk, strength=strength, num_inference_steps=steps,
                  guidance_scale=5.0, width=w8, height=h8,
                  generator=self.torch.Generator("cpu").manual_seed(seed))
        if self.controlnet:
            d = np.nan_to_num(depth if depth is not None else np.ones((H, W)), nan=0, posinf=0)
            inv = np.where(d > 0, 1.0 / np.maximum(d, 1e-3), 0)
            lo, hi = np.percentile(inv[inv > 0], [2, 98]) if (inv > 0).any() else (0, 1)
            inv = np.clip((inv - lo) / max(hi - lo, 1e-6), 0, 1)
            dimg = Image.fromarray((inv * 255).astype(np.uint8)).convert("RGB").resize((w8, h8), Image.BILINEAR)
            kw.update(control_image=dimg, controlnet_conditioning_scale=0.6)
        res = self.pipe(**kw).images[0]
        res = np.asarray(res.resize((W, H), Image.LANCZOS), np.float32) / 255.0
        a = cv2.GaussianBlur(band.astype(np.float32), (0, 0), k / 3)[..., None]
        return render * (1 - a) + res * a, band


# ---------------------------------------------------------------- verification
EDGE_LOSS_LIMIT = 0.1  # max allowed drop in boundary edge recall caused by harmonisation
DRIFT_LIMIT = 6.0  # max allowed median ΔE between harmonised output and the deterministic projection


def _lab(rgb):
    return cv2.cvtColor(np.clip(rgb, 0, 1).astype(np.float32), cv2.COLOR_RGB2LAB)


def verify(original, render, edited):
    """Structure checks: nothing outside the edit changed; structural edges at the edit boundary survive."""
    rep = {}
    outside = ~cv2.dilate(edited.astype(np.uint8), np.ones((25, 25))).astype(bool)
    rep["outside_change"] = float(np.abs(render - original)[outside].mean()) if outside.any() else 0.0
    e0 = cv2.Canny((original * 255).astype(np.uint8), 60, 150) > 0
    e1 = cv2.dilate((cv2.Canny((np.clip(render, 0, 1) * 255).astype(np.uint8), 60, 150) > 0).astype(np.uint8),
                    np.ones((5, 5))) > 0
    boundary = cv2.morphologyEx(edited.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((5, 5))) > 0
    eb = e0 & boundary
    rep["boundary_edge_recall"] = float((eb & e1).sum() / max(1, eb.sum()))
    return rep


def drift_deltaE(projection, final, info, band=None):
    """Median per-pixel ΔE76 between final and projection, per surface, measured only where the
    harmoniser was allowed to change pixels (the seam band); a whole-surface median hides seam damage."""
    a, b = _lab(projection), _lab(final)
    out = {}
    for k, d in info.items():
        m = d["mask"] & band if band is not None else d["mask"]
        out[k] = float(np.median(np.linalg.norm(a[m] - b[m], axis=1))) if m.any() else 0.0
    return out


def sku_recognition(search, render, info, skus):
    """Independent check: crop each edited surface from the render, embed it with SigLIP 2, and
    rank all same-category SKUs by similarity. Pass = the assigned SKU ranks #1."""
    out = {}
    for rid, d in info.items():
        if rid not in skus:  # derived regions (slab edges) are scored with their parent surface
            continue
        m = d["mask"]
        ys, xs = np.nonzero(m)
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        crop = np.where(m[y0:y1, x0:x1, None], render[y0:y1, x0:x1], 0.5)
        # the catalogue is filtered by category, so this has to be the class, never the region id
        ranked = search.rank_image(Image.fromarray((np.clip(crop, 0, 1) * 255).astype(np.uint8)),
                                   kind_of(rid))
        ids = [s.id for s, _ in ranked]
        rank = ids.index(skus[rid].id) + 1 if skus[rid].id in ids else None
        out[rid] = dict(rank=rank, of=len(ids), top=ids[0] if ids else None, pass_=rank == 1)
    return out
