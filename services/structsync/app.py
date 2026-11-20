"""StructuralSync prototype — minimal UI.

Run:  .venv/bin/python app.py   →  http://127.0.0.1:7860
"""
import gradio as gr
import numpy as np
from PIL import Image

from structsync.pipeline import Engine
from structsync.vision import SURFACES
from structsync.viz import COLORS, overlay  # noqa: F401  (re-exported for old imports)

engine = Engine()
KEEP = "— keep original —"


def choices(kind):
    return [KEEP] + [f"{s.id} · {s.name}" for s in engine.catalog if s.category == kind]


def summary(sc):
    rows = ["**Timings:** " + ", ".join(f"{k} {v:.1f}s" for k, v in sc.timings.items()),
            "**Geometry:** " + "; ".join(sc.notes), "", "| surface | planes | ≈ area m² |", "|---|---|---|"]
    for k in SURFACES:
        s = sc.surfaces.get(k)
        if s:
            rows.append(f"| {k} | {len(s.planes)} | {s.area_m2:.2f} |")
    if not sc.surfaces:
        rows.append("| _none detected — use tap-to-correct_ | | |")
    return "\n".join(rows)


def do_analyze(img):
    if img is None:
        raise gr.Error("Upload a photo first")
    sc = engine.analyze(Image.fromarray(img))
    return sc, overlay(sc), summary(sc)


def do_click(sc, kind, mode, evt: gr.SelectData):
    if sc is None:
        raise gr.Error("Analyse a photo first")
    engine.edit_mask(sc, kind, evt.index, add=(mode == "add"))
    return sc, overlay(sc), summary(sc)


def do_search(query, kind):
    if not query:
        return gr.update()
    hits = engine.search.search(query, kind, k=1)
    if not hits:
        return gr.update()
    s = hits[0][0]
    return gr.update(value=f"{s.id} · {s.name}")


def do_render(sc, harmonize, strength, edges, reflections, *picks, material_maps=False, local_tint=False):
    if sc is None:
        raise gr.Error("Analyse a photo first")
    assign = {k: (p.split(" · ")[0] if p and p != KEEP else None) for k, p in zip(SURFACES, picks)}
    if not any(assign.values()):
        raise gr.Error("Pick at least one material")
    out, bom, rep = engine.render(sc, assign, harmonize=harmonize, strength=strength, edges=edges,
                                  reflections=reflections, material_maps=material_maps, local_tint=local_tint)
    before = (sc.full_rgb * 255).astype(np.uint8)  # same size as the full-resolution render
    after = (np.clip(out, 0, 1) * 255).astype(np.uint8)
    total = sum(b["est_cost"] for b in bom)
    table = [[b["surface"], b["sku"], b["product"], b["vendor"], b["range"], b["est_cost"]] for b in bom]
    v = ["**Render:** " + ", ".join(f"{k} {t:.1f}s" for k, t in rep["timings"].items())
         + (" · harmonised" if rep["harmonized"] else " · projection only")
         + (" · slab edges" if edges else "") + (" · reflections" if reflections else ""),
         f"**Verify:** outside-mask change {rep['outside_change']:.4f} (≈0 = untouched) · "
         f"boundary edge recall {rep['boundary_edge_recall']:.2f}"
         + ("" if "drift_deltaE" not in rep else " · harmonise drift ΔE "
            + ", ".join(f"{k} {d:.1f}" for k, d in rep["drift_deltaE"].items()))
         + ("" if "sku_check" not in rep else " · SKU recognised: "
            + ", ".join(f"{k} {'✅' if c['pass_'] else '❌'} (#{c['rank']}/{c['of']})"
                        for k, c in rep["sku_check"].items())),
         f"**Estimated material cost:** ${total:,} (±20 %, verify on site)"]
    if rep.get("note"):
        v.append(f"⚠️ {rep['note']}")
    return (before, after), table, "\n\n".join(v)


with gr.Blocks(title="StructuralSync prototype") as demo:
    gr.Markdown("# StructuralSync — kitchen finish swap (prototype)\n"
                "Upload one photo → **Analyse** → pick catalogue materials → **Render**. "
                "Estimates are visual only — not a structural or measured survey.")
    scene = gr.State()
    with gr.Row():
        with gr.Column(scale=1):
            inp = gr.Image(label="1 · Room photo", type="numpy")
            btn_an = gr.Button("Analyse photo", variant="primary")
            info = gr.Markdown()
        with gr.Column(scale=1):
            ov = gr.Image(label="2 · Detected surfaces (click to correct)", interactive=False)
            with gr.Row():
                fix_kind = gr.Dropdown(SURFACES, value="countertop", label="Surface to correct")
                fix_mode = gr.Radio(["add", "remove"], value="add", label="Click action")
    gr.Markdown("### 3 · Materials (fixed catalogue)")
    with gr.Row():
        q = gr.Textbox(label="Search catalogue", placeholder="e.g. dark marble, navy, light oak")
        q_kind = gr.Dropdown(SURFACES, value="countertop", label="for surface")
        btn_q = gr.Button("Find")
    with gr.Row():
        picks = [gr.Dropdown(choices(k), value=KEEP, label=k) for k in SURFACES]
    with gr.Row():
        harm = gr.Checkbox(label="Harmonise seams with SDXL (slow on Mac, ~7 GB download first time)")
        strength = gr.Slider(0.1, 0.5, 0.3, step=0.05, label="Harmonise strength")
        edges = gr.Checkbox(value=True, label="Countertop front edges")
        refl = gr.Checkbox(value=False, label="Mirror reflections on polished stone (experimental)")
        btn_r = gr.Button("Render", variant="primary")
    slider = gr.ImageSlider(label="4 · Before / after")
    bom = gr.Dataframe(headers=["surface", "SKU", "product", "vendor", "≈ m² range", "est. $"],
                       label="Bill of materials")
    rep = gr.Markdown()

    btn_an.click(do_analyze, inp, [scene, ov, info])
    ov.select(do_click, [scene, fix_kind, fix_mode], [scene, ov, info])

    def route_search(query, kind):
        upd = do_search(query, kind)
        return [upd if k == kind else gr.update() for k in SURFACES]

    btn_q.click(route_search, [q, q_kind], picks)
    btn_r.click(do_render, [scene, harm, strength, edges, refl, *picks], [slider, bom, rep])

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1", help="0.0.0.0 to expose on a GPU server")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true", help="public gradio.live tunnel")
    ap.add_argument("--preload", action="store_true", help="load all models at start-up")
    a = ap.parse_args()
    if a.preload:
        engine.preload()
    demo.launch(server_name=a.host, server_port=a.port, share=a.share)
