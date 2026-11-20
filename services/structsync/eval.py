"""Batch evaluation: run the full pipeline over a folder of photos and write a report.

    python eval.py photos/ [--harmonize] [--out outputs/eval]

For every image: analyse once, then render each catalogue SKU on every detected surface
(one SKU per surface per render, round-robin). Writes per-image overlay + render grids and
results.csv (timings, areas, structure checks, harmonise drift, SKU recognition rank).
"""
import argparse
import csv
import itertools
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from structsync.pipeline import Engine
from structsync.vision import SURFACES
from structsync.viz import overlay


def grid(images, cols=3, w=512):
    ims = [Image.fromarray(i).resize((w, int(w * i.shape[0] / i.shape[1]))) for i in images]
    h = max(i.height for i in ims)
    rows = (len(ims) + cols - 1) // cols
    g = Image.new("RGB", (cols * w, rows * h), "white")
    for k, im in enumerate(ims):
        g.paste(im, ((k % cols) * w, (k // cols) * h))
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--out", default="outputs/eval")
    ap.add_argument("--harmonize", action="store_true")
    ap.add_argument("--renders", type=int, default=4, help="material combinations per photo")
    ap.add_argument("--no-edges", action="store_true", help="don't re-texture countertop front edges")
    ap.add_argument("--reflections", action="store_true", help="experimental mirror reflections")
    ap.add_argument("--no-grain", action="store_true", help="don't add the photo's sensor noise")
    ap.add_argument("--material-maps", action="store_true", help="use SKU normal/roughness maps")
    ap.add_argument("--local-tint", action="store_true", help="per-surface neutral-neighbour tint")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    eng = Engine(edges=not a.no_edges, reflections=a.reflections, grain=not a.no_grain,
                 material_maps=a.material_maps, local_tint=a.local_tint)
    switches = dict(edges=eng.edges, reflections=eng.reflections, grain=eng.grain,
                    material_maps=eng.material_maps, local_tint=eng.local_tint)
    t = time.time()
    eng.preload(harmonizer=a.harmonize)
    print(f"device={eng.device} model load {time.time() - t:.1f}s")

    by_cat = {k: [s.id for s in eng.catalog if s.category == k] for k in SURFACES}
    paths = sorted(p for p in Path(a.folder).iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"))
    rows = []
    for p in paths:
        sc = eng.analyze(Image.open(p))
        found = [k for k in SURFACES if k in sc.surfaces and sc.surfaces[k].planes and by_cat[k]]
        tiles = [(sc.rgb * 255).astype(np.uint8), overlay(sc)]
        cycles = {k: itertools.cycle(by_cat[k]) for k in found}
        for r in range(a.renders if found else 0):
            assign = {k: next(cycles[k]) for k in found}
            img, bom, rep = eng.render(sc, assign, harmonize=a.harmonize)
            tiles.append((np.clip(img, 0, 1) * 255).astype(np.uint8))
            rows.append(dict(
                image=p.name, render=r, **switches, **{f"t_{k}": round(v, 2) for k, v in sc.timings.items()},
                **{f"t_{k}": round(v, 2) for k, v in rep["timings"].items()},
                surfaces=";".join(found),
                areas=";".join(f"{k}={sc.surfaces[k].area_m2:.2f}" for k in found),
                scale_note=sc.notes[-1], assign=json.dumps(assign),
                outside_change=round(rep["outside_change"], 5),
                edge_recall=round(rep["boundary_edge_recall"], 3),
                harmonized=rep["harmonized"], drift=json.dumps(rep.get("drift_deltaE", {})),
                sku_pass=sum(c["pass_"] for c in rep.get("sku_check", {}).values()),
                sku_total=len(rep.get("sku_check", {})), note=rep.get("note", "")))
        grid(tiles).save(out / f"{p.stem}_grid.jpg", quality=90)
        print(f"{p.name}: surfaces={found} analyse={sum(sc.timings.values()):.1f}s")

    if rows:
        with open(out / "results.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}))
            w.writeheader()
            w.writerows(rows)
        sp, st = sum(r["sku_pass"] for r in rows), sum(r["sku_total"] for r in rows)
        an = [sum(v for k, v in r.items() if k in ("t_segment", "t_geometry", "t_surfaces"))
              for r in rows if r["render"] == 0]
        sw = [r["t_projection"] + r.get("t_harmonize", 0) for r in rows]
        print(f"\n{len(paths)} photos, {len(rows)} renders | analyse median {np.median(an):.1f}s | "
              f"swap median {np.median(sw):.2f}s p90 {np.percentile(sw, 90):.2f}s | "
              f"SKU recognised {sp}/{st} | outside-change max {max(r['outside_change'] for r in rows):.4f}")
        print(f"report: {out / 'results.csv'}")


if __name__ == "__main__":
    main()
