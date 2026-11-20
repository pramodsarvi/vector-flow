"""Shared visualisation helpers (no UI framework imports, so scripts and servers can use them)."""
import cv2
import numpy as np

from .regions import RID_SEP, kind_of

COLORS = {"countertop": (255, 80, 80), "cabinet": (80, 160, 255), "backsplash": (255, 200, 0),
          "floor": (120, 220, 120), "wall": (200, 120, 255), "occluder": (90, 90, 90),
          "ceiling": (120, 200, 255)}
FALLBACK = (160, 160, 160)


def region_color(rid: str):
    """One colour per class, lightened per region index so neighbouring regions of the same class
    stay apart on the overlay (a run of cabinets is the case that matters)."""
    base = np.array(COLORS.get(kind_of(rid), FALLBACK), np.float32)
    _, _, tail = rid.partition(RID_SEP)
    n = int(tail) if tail.isdigit() else 0
    # alternate darker/lighter around the base colour; 4 steps before it repeats
    f = (1.0, 0.72, 1.0, 0.86)[n % 4]
    if n % 4 == 2:
        base = base + (255 - base) * 0.35
    return tuple(float(x) for x in np.clip(base * f, 0, 255))


def overlay(sc):
    """Analysis-resolution photo with each detected region tinted and outlined."""
    img = (sc.rgb * 255).astype(np.uint8).copy()
    col = img.copy()
    for k, m in sc.masks.items():
        col[m] = region_color(k)
    out = cv2.addWeighted(img, 0.55, col, 0.45, 0)
    for k, m in sc.masks.items():
        cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cs, -1, region_color(k), 2)
    return out
