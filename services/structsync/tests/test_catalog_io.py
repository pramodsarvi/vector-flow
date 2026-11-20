"""Bulk catalogue import/export (numpy/PIL only, no models)."""
import io
import json
import shutil
import zipfile

import numpy as np
import pytest
from conftest import ROOT
from PIL import Image

import structsync.catalog as C
import structsync.catalog_io as CIO
from structsync.vision import SURFACES

CATS = SURFACES          # the importer accepts exactly the surface classes


@pytest.fixture
def cat(tmp_path):
    root = tmp_path / "catalog"
    shutil.copytree(ROOT / "tests" / "fixtures" / "catalog", root)
    (root / "siglip_embeddings.npy").unlink(missing_ok=True)
    return root


def snapshot(root):
    """Directory listing + catalog.json bytes (ignores nothing): used to prove 'nothing changed'."""
    files = sorted(str(p.relative_to(root)) for p in root.rglob("*"))
    return files, (root / "catalog.json").read_bytes()


def png(color=(120, 80, 40), size=96, mode="RGB"):
    b = io.BytesIO()
    Image.new(mode, (size, size), color if mode == "RGB" else 128).save(b, "PNG")
    return b.getvalue()


def make_zip(path, members: dict):
    with zipfile.ZipFile(path, "w") as z:
        for name, data in members.items():
            z.writestr(name, data)
    return path


def load(root):
    old = C.ROOT
    C.ROOT = root
    try:
        return C.load_catalog()
    finally:
        C.ROOT = old


def test_export_then_replace_import_round_trips(cat, tmp_path):
    out = CIO.export_zip(cat, tmp_path / "export.zip")
    other = tmp_path / "other"
    other.mkdir()
    (other / "catalog.json").write_text("[]")
    plan = CIO.plan_import(out, other, CATS, "replace")
    assert plan.errors == [] and len(plan.added) == len(load(cat))
    CIO.apply_import(out, other, plan)
    assert json.loads((other / "catalog.json").read_text()) == json.loads((cat / "catalog.json").read_text())
    for s in load(other):
        a = np.asarray(Image.open(cat / s.id / "albedo.png").convert("RGB"))
        b = np.asarray(Image.open(other / s.id / "albedo.png").convert("RGB"))
        assert np.array_equal(a, b), s.id
    assert not list(other.glob(".import-*"))


def test_supplier_csv_messy_zip_merges(cat, tmp_path):
    n_before = len(load(cat))
    csv_text = "﻿ ID ; Name;CATEGORY ; vendor;price;tile_w;tile_h;gloss;extra\n" \
               "CT-NEW-01;New Quartz;Countertop;Acme;350,5;1.2;1.2;0.6;x\n" \
               "CB-NEW-02;New Paint;cabinet;Acme;40;0.5;0.5;;y\n"
    z = make_zip(tmp_path / "supplier.zip", {
        "Supplier Pack/catalog.csv": csv_text.encode("utf-8"),
        "Supplier Pack/CT-NEW-01/ALBEDO.JPG": _jpg(),
        "Supplier Pack\\CB-NEW-02\\albedo.png": png((30, 40, 90)),
        "__MACOSX/Supplier Pack/._catalog.csv": b"junk",
        "Supplier Pack/.DS_Store": b"junk",
    })
    plan = CIO.plan_import(z, cat, CATS, "merge")
    assert plan.errors == [], plan.errors
    assert sorted(plan.added) == ["CB-NEW-02", "CT-NEW-01"] and plan.removed == []
    assert any("ignored columns" in w for w in plan.warnings)
    CIO.apply_import(z, cat, plan)
    skus = {s.id: s for s in load(cat)}
    assert len(skus) == n_before + 2
    assert skus["CT-NEW-01"].price == 350.5 and skus["CT-NEW-01"].category == "countertop"
    assert skus["CB-NEW-02"].gloss == 0.3                               # default
    assert skus["CT-NEW-01"].albedo.shape[2] == 3


def _jpg():
    b = io.BytesIO()
    Image.new("RGB", (128, 128), (200, 190, 180)).save(b, "JPEG")
    return b.getvalue()


def test_any_bad_row_changes_nothing_and_reports_rows(cat, tmp_path):
    before = snapshot(cat)
    rows = [dict(id="CT-OK-01", name="ok", category="countertop", vendor="", price=1, tile_w=1, tile_h=1, gloss=0.5),
            dict(id="bad id!", name="x", category="countertop", price=1, tile_w=1, tile_h=1, gloss=0.5),
            dict(id="CT-NOTEX-02", name="no texture", category="countertop", price=1, tile_w=1, tile_h=1, gloss=0.5),
            dict(id="CT-BIG-03", name="x", category="roof", price=-5, tile_w=0, tile_h=1, gloss=2)]
    z = make_zip(tmp_path / "bad.zip", {"catalog.json": json.dumps(rows), "CT-OK-01/albedo.png": png(),
                                        "CT-BIG-03/albedo.png": b"not an image"})
    plan = CIO.plan_import(z, cat, CATS, "merge")
    text = "\n".join(plan.errors)
    assert "row 2" in text and "row 3" in text and "row 4" in text and "not a readable image" in text
    with pytest.raises(CIO.CatalogImportError):
        CIO.apply_import(z, cat, plan)
    assert snapshot(cat) == before


def test_crash_mid_apply_leaves_loadable_catalogue_and_recovers(cat, tmp_path):
    rows = json.loads((cat / "catalog.json").read_text())
    rows[0]["price"] = 999
    z = make_zip(tmp_path / "upd.zip", {"catalog.json": json.dumps(rows[:3]),
                                        f"{rows[0]['id']}/albedo.png": png((10, 200, 10)),
                                        f"{rows[1]['id']}/albedo.png": png((200, 10, 10))})
    plan = CIO.plan_import(z, cat, CATS, "replace")
    with pytest.raises(RuntimeError):
        CIO.apply_import(z, cat, plan, _fail_after_swaps=1)
    skus = load(cat)                                           # still loadable mid-crash
    assert all((s.path / "albedo.png").exists() for s in skus)
    assert CIO.recover(cat) and not list(cat.glob(".import-*"))
    after = load(cat)
    assert [s.id for s in after] == [r["id"] for r in rows[:3]] and after[0].price == 999
    assert all((s.path / "albedo.png").exists() for s in after)
    assert not (cat / rows[5]["id"]).exists()                  # removed SKU folders deleted


def test_unfinished_staging_is_discarded(cat):
    before = snapshot(cat)
    (cat / ".import-deadbeef").mkdir()
    (cat / ".import-deadbeef" / "CT-X-01").mkdir()
    assert "discarded" in CIO.recover(cat)[0]
    assert snapshot(cat) == before


def test_csv_only_merge_updates_price_and_keeps_texture(cat, tmp_path):
    sid = "CT-CAL-01"
    n_before = len(load(cat))
    tex = (cat / sid / "albedo.png").read_bytes()
    csv_text = ("id,name,category,vendor,price,tile_w,tile_h,gloss\n"
                "CT-CAL-01,Calacatta White Quartz,countertop,StoneCo,499,1.6,1.6,0.6\n")
    z = make_zip(tmp_path / "prices.zip", {"catalog.csv": csv_text})
    plan = CIO.plan_import(z, cat, CATS, "merge")
    assert plan.errors == [] and plan.updated == [sid]
    CIO.apply_import(z, cat, plan)
    assert next(s for s in load(cat) if s.id == sid).price == 499
    assert (cat / sid / "albedo.png").read_bytes() == tex
    assert len(load(cat)) == n_before          # a CSV-only merge adds and removes nothing


def test_new_albedo_without_maps_drops_old_maps(cat, tmp_path):
    sid = "BS-SUB-01"
    Image.new("RGB", (64, 64), (128, 128, 255)).save(cat / sid / "normal.png")
    rows = [e for e in json.loads((cat / "catalog.json").read_text()) if e["id"] == sid]
    z = make_zip(tmp_path / "tex.zip", {"catalog.json": json.dumps(rows), f"{sid}/albedo.png": png()})
    CIO.apply_import(z, cat, CIO.plan_import(z, cat, CATS, "merge"))
    assert not (cat / sid / "normal.png").exists()
    z2 = make_zip(tmp_path / "maps.zip", {"catalog.json": json.dumps(rows), f"{sid}/roughness.png": png(mode="L")})
    CIO.apply_import(z2, cat, CIO.plan_import(z2, cat, CATS, "merge"))   # maps-only update keeps texture
    assert (cat / sid / "albedo.png").exists() and (cat / sid / "roughness.png").exists()


def test_rejects_case_clash_duplicates_and_oversize(cat, tmp_path, monkeypatch):
    base = dict(name="x", category="countertop", vendor="", price=1, tile_w=1, tile_h=1, gloss=0.5)
    rows = [dict(base, id="ct-cal-01"), dict(base, id="CT-A-01"), dict(base, id="ct-a-01")]
    z = make_zip(tmp_path / "dup.zip", {"catalog.json": json.dumps(rows), "CT-A-01/albedo.png": png()})
    text = "\n".join(CIO.plan_import(z, cat, CATS, "merge").errors)
    assert "differs only by case" in text and "duplicate id" in text
    monkeypatch.setattr(CIO, "MAX_MEMBER_BYTES", 100)
    z = make_zip(tmp_path / "big.zip", {"catalog.json": json.dumps([dict(base, id="CT-B-01")]),
                                        "CT-B-01/albedo.png": png(size=256)})
    assert any("larger than" in e for e in CIO.plan_import(z, cat, CATS, "merge").errors)
    assert CIO.plan_import(tmp_path / "missing.zip", cat, CATS).errors == ["not a valid ZIP file"]
