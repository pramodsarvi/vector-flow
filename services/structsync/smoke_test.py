"""Headless end-to-end run: analyse a sample photo, render two material sets, save outputs."""
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from structsync.pipeline import Engine  # noqa: E402
from structsync.viz import overlay  # noqa: E402

out_dir = Path(__file__).parent / "outputs"
out_dir.mkdir(exist_ok=True)
img = Image.open(next((a for a in sys.argv[1:] if not a.startswith("--")), "samples/kitchen1_ccby.jpg"))

eng = Engine(edges="--no-edges" not in sys.argv, reflections="--reflections" in sys.argv,
             grain="--no-grain" not in sys.argv, material_maps="--material-maps" in sys.argv,
             local_tint="--local-tint" in sys.argv)
t = time.time()
sc = eng.analyze(img)
print(f"analyse {time.time() - t:.1f}s", sc.timings, sc.notes)
for k, s in sc.surfaces.items():
    print(f"  {k:11s} px={s.mask.sum():7d} planes={len(s.planes)} area≈{s.area_m2:.2f} m²")
Image.fromarray(overlay(sc)).save(out_dir / "overlay.png")

for name, assign in {
    "white_quartz_navy": {"countertop": "CT-CAL-01", "cabinet": "CB-NAV-01"},
    "oak_sage_subway": {"countertop": "CT-OAK-04", "cabinet": "CB-SAG-02", "backsplash": "BS-SUB-01"},
}.items():
    out, bom, rep = eng.render(sc, assign, harmonize="--harmonize" in sys.argv)
    Image.fromarray((np.clip(out, 0, 1) * 255).astype(np.uint8)).save(out_dir / f"{name}.png")
    print(name, rep, bom)

print("search 'dark marble' →", [(s.id, round(v, 3)) for s, v in eng.search.search("dark marble", "countertop")])
