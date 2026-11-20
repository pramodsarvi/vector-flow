"""Web API end to end on a synthetic room: real geometry fitting, rendering and persistence; the
detector (SAM click), SigLIP search and harmoniser are stubbed, so no model is ever loaded."""
import importlib
import io
import json
import os
import shutil
import sys
import time
import zipfile

import numpy as np
import pytest
from conftest import ROOT, photo, room_state, two_wall_room
from PIL import Image

import structsync.catalog as C
import structsync.pipeline as P


class FakeSearch:
    def __init__(self, skus, device="cpu"):
        self.skus = skus

    def rank_image(self, img, category=None):
        same = [s for s in self.skus if not category or s.category == category]
        return [(s, 1.0 - 0.1 * i) for i, s in enumerate(same)]

    def search(self, text, category=None, k=3):
        same = [s for s in self.skus if not category or s.category == category]
        return [(s, 0.5) for s in same][:k]


class FakeSeg:
    def mask_from_point(self, img, xy, positive=True):
        w, h = img.size
        yy, xx = np.mgrid[0:h, 0:w]
        return (xx - xy[0]) ** 2 + (yy - xy[1]) ** 2 < 25 ** 2


def png_bytes(img):
    b = io.BytesIO()
    img.save(b, "PNG")
    return b.getvalue()


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    from fastapi.testclient import TestClient
    data, cat = tmp_path_factory.mktemp("data"), tmp_path_factory.mktemp("catalog")
    shutil.copytree(ROOT / "tests" / "fixtures" / "catalog", cat, dirs_exist_ok=True)
    (cat / "siglip_embeddings.npy").unlink(missing_ok=True)
    saved = (C.ROOT, P.CatalogSearch, os.environ.get("STRUCTSYNC_DATA"))
    C.ROOT, P.CatalogSearch = cat, FakeSearch
    os.environ["STRUCTSYNC_DATA"] = str(data)
    sys.modules.pop("web.server", None)
    server = importlib.import_module("web.server")
    eng = server.engine
    eng._seg = FakeSeg()
    eng._harm_error = "stub: no harmoniser in tests"
    eng.analyze = lambda img: eng.restore(img, room_state(*eng.prepare(img)[0].size), ["synthetic"], {"segment": 0.0})
    try:
        with TestClient(server.app) as c:
            c.server = server
            yield c
    finally:
        C.ROOT, P.CatalogSearch = saved[0], saved[1]
        if saved[2] is None:
            os.environ.pop("STRUCTSYNC_DATA", None)
        else:
            os.environ["STRUCTSYNC_DATA"] = saved[2]
        sys.modules.pop("web.server", None)


@pytest.fixture(scope="module")
def project(client):
    r = client.post("/api/projects", data={"name": "test room"},
                    files={"photo": ("room.png", png_bytes(photo()), "image/png")})
    assert r.status_code == 200, r.text
    pid = r.json()["id"]
    for _ in range(300):
        p = client.get(f"/api/projects/{pid}").json()
        if p["status"] != "analyzing":
            break
        time.sleep(0.05)
    assert p["status"] == "ready", p
    return p


def area(p, kind):
    return next((s["area_m2"] for s in p["surfaces"] if s["kind"] == kind), 0.0)


@pytest.fixture(scope="module")
def split_project(client):
    """A project whose analysis actually splits classes into regions, so the API can be exercised
    against more than one region of the same class (the module's main room has exactly one each)."""
    eng = client.server.engine
    saved = eng.analyze

    def analyze(img):
        sc = eng.restore(img, two_wall_room(*eng.prepare(img)[0].size), ["synthetic"], {})
        eng._instance(sc)
        return sc

    eng.analyze = analyze
    try:
        r = client.post("/api/projects", data={"name": "two walls"},
                        files={"photo": ("room.png", png_bytes(photo(480, 360)), "image/png")})
        assert r.status_code == 200, r.text
        pid = r.json()["id"]
        for _ in range(300):
            p = client.get(f"/api/projects/{pid}").json()
            if p["status"] != "analyzing":
                break
            time.sleep(0.05)
        assert p["status"] == "ready", p
    finally:
        # restore before yielding: this fixture is module-scoped, so leaving the override in place
        # would hand the two-wall room to every project created after it
        eng.analyze = saved
    return p


def test_analysis_reports_addressable_regions(client, split_project):
    regions = split_project["regions"]
    ids = [r["id"] for r in regions]
    assert sorted(i for i in ids if i.startswith("wall")) == ["wall#0", "wall#1"], ids
    assert [i for i in ids if i.startswith("floor")] == ["floor#0"]
    for r in regions:
        assert r["kind"] == r["id"].split("#")[0]
        assert r["area_m2"] > 0 and len(r["color"]) == 3
    # the per-class summary keeps its old shape, with the regions counted and areas summed
    wall = next(s for s in split_project["surfaces"] if s["kind"] == "wall")
    assert wall["regions"] == 2
    assert wall["area_m2"] == pytest.approx(
        sum(r["area_m2"] for r in regions if r["kind"] == "wall"), abs=0.02)


def test_two_walls_take_different_materials(client, split_project):
    pid = split_project["id"]
    r = client.post(f"/api/projects/{pid}/render",
                    json={"assign": {"wall#0": "WL-GRG-01", "wall#1": "CT-CAL-01"}})
    assert r.status_code == 200, r.text
    rec = r.json()
    assert {(b["sku"], tuple(b["regions"])) for b in rec["bom"]} == \
        {("WL-GRG-01", ("wall#0",)), ("CT-CAL-01", ("wall#1",))}
    assert rec["report"]["outside_change"] < 1e-4


def test_a_bare_class_still_covers_every_region_of_it(client, split_project):
    """The documented shape, and what an older client sends."""
    pid = split_project["id"]
    rec = client.post(f"/api/projects/{pid}/render", json={"assign": {"wall": "WL-GRG-01"}}).json()
    assert len(rec["bom"]) == 1
    assert sorted(rec["bom"][0]["regions"]) == ["wall#0", "wall#1"]


def test_region_and_class_assignments_key_the_cache_differently(client, split_project):
    """One wall restyled is not the same render as both of them."""
    pid = split_project["id"]
    one = client.post(f"/api/projects/{pid}/render", json={"assign": {"wall#0": "WL-GRG-01"}}).json()
    both = client.post(f"/api/projects/{pid}/render", json={"assign": {"wall": "WL-GRG-01"}}).json()
    assert one["key"] != both["key"]
    assert one["id"] != both["id"]


def test_correcting_one_region_leaves_its_neighbour_alone(client, split_project):
    pid = split_project["id"]
    before = {r["id"]: r["area_m2"] for r in client.get(f"/api/projects/{pid}").json()["regions"]}
    # wall#1 is the smaller of the two (ids run largest first), i.e. the left wall at x = −1.5 m,
    # which in a 480-wide frame is everything left of x ≈ 132. (15, 150) sits well inside it and
    # clear of both the floor and the counter band.
    r = client.post(f"/api/projects/{pid}/correct",
                    json={"kind": "wall", "region": "wall#1", "x": 15, "y": 150, "mode": "remove"})
    assert r.status_code == 200, r.text
    after = {x["id"]: x["area_m2"] for x in r.json()["regions"]}
    assert after["wall#1"] < before["wall#1"]
    assert after["wall#0"] == pytest.approx(before["wall#0"], abs=0.02)


def test_bad_region_is_rejected(client, split_project):
    pid = split_project["id"]
    assert client.post(f"/api/projects/{pid}/correct",
                       json={"kind": "wall", "region": "sofa#0", "x": 10, "y": 10}).status_code == 400


def test_ceiling_is_a_selectable_surface(client, split_project):
    """Ceiling is a first-class finish surface: detected from geometry like any other, offered in
    the config, carried in the catalogue and billed. Read-only, so it leaves the shared project
    alone — tap-to-correct on a ceiling is covered at engine level in test_regions."""
    cfg = client.get("/api/config").json()
    assert "ceiling" in cfg["surfaces"] and "ceiling" in cfg["colors"]
    ceilings = [r for r in split_project["regions"] if r["kind"] == "ceiling"]
    assert [r["id"] for r in ceilings] == ["ceiling#0"], split_project["regions"]
    assert ceilings[0]["area_m2"] > 1.0
    sku = next(s for s in client.get("/api/catalog").json() if s["category"] == "ceiling")
    r = client.post(f"/api/projects/{split_project['id']}/render",
                    json={"assign": {"ceiling": sku["id"]}})
    assert r.status_code == 200, r.text
    rec = r.json()
    assert [b["surface"] for b in rec["bom"]] == ["ceiling"]
    assert rec["bom"][0]["regions"] == ["ceiling#0"]
    assert rec["report"]["outside_change"] < 1e-4


def test_config_health_catalog(client):
    assert client.get("/api/config").json()["switches"]["material_maps"] is False
    h = client.get("/api/health").json()
    assert h["harmonizer"]["loaded"] is False
    cat = client.get("/api/catalog").json()
    assert len(cat) == 15 and {"has_normal", "has_roughness"} <= set(cat[0])
    assert client.get("/api/catalog/search", params={"q": "oak", "category": "floor"}).status_code == 200


def test_token_required_when_configured(client, monkeypatch):
    monkeypatch.setattr(client.server, "TOKEN", "s3cret")
    assert client.get("/api/config").status_code == 401
    assert client.get("/api/config", headers={"Authorization": "Bearer s3cret"}).status_code == 200
    assert client.get("/api/config?token=s3cret").status_code == 200


def test_rejects_bad_uploads(client):
    assert client.post("/api/projects", files={"photo": ("x.jpg", b"not an image")}).status_code == 400
    tiny = png_bytes(Image.new("RGB", (100, 80)))
    assert client.post("/api/projects", files={"photo": ("x.png", tiny)}).status_code == 400


def test_project_analysis_and_images(client, project):
    assert area(project, "floor") > 0 and area(project, "wall") > 0 and area(project, "cabinet") > 0
    pid = project["id"]
    assert client.get(f"/api/projects/{pid}/overlay").content[:4] == b"\x89PNG"
    im = Image.open(io.BytesIO(client.get(f"/api/projects/{pid}/photo").content))
    assert im.size == (400, 300)


def test_tap_to_correct(client, project):
    pid = project["id"]
    before = area(project, "cabinet")
    r = client.post(f"/api/projects/{pid}/correct", json={"kind": "cabinet", "x": 100, "y": 120, "mode": "remove"})
    assert r.status_code == 200 and area(r.json(), "cabinet") < before
    r = client.post(f"/api/projects/{pid}/correct", json={"kind": "cabinet", "x": 100, "y": 120, "mode": "add"})
    assert r.status_code == 200 and area(r.json(), "cabinet") > 0
    assert client.post(f"/api/projects/{pid}/correct",
                       json={"kind": "cabinet", "x": 5000, "y": 1, "mode": "add"}).status_code == 400
    # ceiling is a real surface class now, so an unknown kind has to be something else entirely
    assert client.post(f"/api/projects/{pid}/correct",
                       json={"kind": "sofa", "x": 1, "y": 1, "mode": "add"}).status_code == 400


@pytest.mark.parametrize("opts", [{}, {"edges": False}, {"grain": False}, {"reflections": True},
                                  {"material_maps": True}, {"local_tint": True}, {"harmonize": True}])
def test_render_switches(client, project, opts):
    pid = project["id"]
    r = client.post(f"/api/projects/{pid}/render",
                    json={"assign": {"floor": "FL-OAK-01", "wall": "WL-GRG-01"}, **opts})
    assert r.status_code == 200, r.text
    rec = r.json()
    assert rec["report"]["outside_change"] < 1e-4
    assert {b["surface"] for b in rec["bom"]} == {"floor", "wall"}
    if opts.get("harmonize"):
        assert "harmoniser unavailable" in rec["report"]["note"]
    img = client.get(f"/api/projects/{pid}/renders/{rec['id']}.jpg")
    assert img.status_code == 200 and Image.open(io.BytesIO(img.content)).size == (400, 300)


def test_render_validation(client, project):
    pid = project["id"]
    assert client.post(f"/api/projects/{pid}/render", json={"assign": {"floor": "NOPE"}}).status_code == 400
    assert client.post(f"/api/projects/{pid}/render", json={"assign": {}}).status_code == 400


def test_catalogue_crud_and_maps(client):
    n_before = len(client.get("/api/catalog").json())
    tex = png_bytes(photo(128, 128, seed=5))
    form = dict(id="CT-TEST-01", name="Test Stone", category="countertop", vendor="T", price="10",
                tile_w="1", tile_h="1", gloss="0.5")
    bad = client.post("/api/catalog", data=dict(form, id="../evil"), files={"albedo": ("t.png", tex)})
    assert bad.status_code == 400
    r = client.post("/api/catalog", data=form, files={"albedo": ("t.png", tex), "normal": ("n.png", tex)})
    assert r.status_code == 200, r.text
    assert client.post("/api/catalog", data=form, files={"albedo": ("t.png", tex)}).status_code == 409
    assert client.put("/api/catalog/CT-TEST-01", json={"gloss": 3}).status_code == 400
    assert client.put("/api/catalog/CT-TEST-01", json={"price": 20}).json()["price"] == 20
    sku = next(s for s in client.get("/api/catalog").json() if s["id"] == "CT-TEST-01")
    assert sku["has_normal"] and not sku["has_roughness"]
    r = client.post("/api/catalog/CT-TEST-01/maps", files={"roughness": ("r.png", tex)})
    assert r.status_code == 200 and r.json()["has_roughness"]
    assert client.get("/api/catalog/CT-TEST-01/map/normal").status_code == 200
    assert client.delete("/api/catalog/CT-TEST-01/maps/normal").json()["has_normal"] is False
    assert client.post("/api/catalog/CT-TEST-01/albedo", files={"albedo": ("t.png", tex)}).status_code == 200
    assert client.delete("/api/catalog/CT-TEST-01").status_code == 200
    assert len(client.get("/api/catalog").json()) == n_before


def test_persistence_and_deletes(client, project):
    pid = project["id"]
    srv = client.server
    before = client.get(f"/api/projects/{pid}").json()
    srv._scenes.clear()                                   # simulate a restart: scene must come from disk
    r = client.post(f"/api/projects/{pid}/render", json={"assign": {"wall": "WL-GRG-01"}})
    assert r.status_code == 200
    after = client.get(f"/api/projects/{pid}").json()
    assert area(after, "cabinet") == area(before, "cabinet")
    rid = after["renders"][0]["id"]
    assert client.delete(f"/api/projects/{pid}/renders/{rid}").status_code == 200
    assert client.get(f"/api/projects/{pid}/renders/{rid}.jpg").status_code == 404
    assert client.get("/api/projects/..%2F..%2Fetc").status_code == 404
    assert client.patch(f"/api/projects/{pid}", json={"name": "renamed"}).json()["name"] == "renamed"
    assert client.delete(f"/api/projects/{pid}").status_code == 200
    assert all(p["id"] != pid for p in client.get("/api/projects").json())


def test_catalogue_export_and_bulk_import(client, tmp_path):
    exp = client.get("/api/catalog/export")
    assert exp.status_code == 200 and exp.headers["content-type"] == "application/zip"
    with zipfile.ZipFile(io.BytesIO(exp.content)) as z:
        rows = json.loads(z.read("catalog.json"))
    assert len(rows) == len(client.get("/api/catalog").json())
    rows[0]["price"] = 777
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("catalog.json", json.dumps(rows[:1]))
    files = {"archive": ("c.zip", buf.getvalue(), "application/zip")}
    dry = client.post("/api/catalog/import", data={"mode": "merge", "dry_run": "true"}, files=files).json()
    assert dry["ok"] and not dry["applied"] and dry["updated"] == [rows[0]["id"]]
    assert next(s for s in client.get("/api/catalog").json() if s["id"] == rows[0]["id"])["price"] != 777
    done = client.post("/api/catalog/import", data={"mode": "merge", "dry_run": "false"}, files=files).json()
    assert done["applied"]
    assert next(s for s in client.get("/api/catalog").json() if s["id"] == rows[0]["id"])["price"] == 777
    bad = client.post("/api/catalog/import", data={"mode": "merge", "dry_run": "false"},
                      files={"archive": ("x.zip", b"not a zip", "application/zip")})
    assert bad.status_code == 422 and bad.json()["errors"] == ["not a valid ZIP file"]


# ------------------------------------------------------------------ caching
class Counter:
    def __init__(self, fn):
        self.fn, self.n = fn, 0

    def __call__(self, *a, **k):
        self.n += 1
        return self.fn(*a, **k)


@pytest.fixture
def counted(client, monkeypatch):
    eng = client.server.engine
    an, rn = Counter(eng.analyze), Counter(eng.render)
    monkeypatch.setattr(eng, "analyze", an)
    monkeypatch.setattr(eng, "render", rn)
    return an, rn


def new_project(client, seed, name="cache"):
    r = client.post("/api/projects", data={"name": name},
                    files={"photo": ("p.png", png_bytes(photo(seed=seed)), "image/png")})
    pid = r.json()["id"]
    for _ in range(300):
        p = client.get(f"/api/projects/{pid}").json()
        if p["status"] != "analyzing":
            return p
        time.sleep(0.05)
    raise AssertionError("analysis did not finish")


def render(client, pid, **opts):
    r = client.post(f"/api/projects/{pid}/render", json={"assign": {"floor": "FL-OAK-01", "wall": "WL-GRG-01"}, **opts})
    assert r.status_code == 200, r.text
    return r.json()


def test_same_photo_reuses_analysis_reanalyse_recomputes(client, counted):
    an, _ = counted
    a = new_project(client, seed=21)
    b = new_project(client, seed=21)
    assert an.n == 1 and b["analysis_cached"] and not a["analysis_cached"]
    assert b["surfaces"] == a["surfaces"] and b["scene_rev"] == a["scene_rev"]
    client.post(f"/api/projects/{b['id']}/reanalyze")
    for _ in range(300):
        if client.get(f"/api/projects/{b['id']}").json()["status"] == "ready":
            break
        time.sleep(0.05)
    assert an.n == 2


def test_same_style_is_not_recomputed(client, counted):
    _, rn = counted
    p = new_project(client, seed=22)
    r1 = render(client, p["id"])
    r2 = render(client, p["id"])
    assert rn.n == 1 and r2["id"] == r1["id"] and r2["cached"]
    assert len(client.get(f"/api/projects/{p['id']}").json()["renders"]) == 1
    render(client, p["id"], grain=False)                                     # different option → render
    assert rn.n == 2


def test_cache_invalidation(client, counted):
    _, rn = counted
    p = new_project(client, seed=23)
    pid = p["id"]
    render(client, pid)
    client.post(f"/api/projects/{pid}/correct", json={"kind": "wall", "x": 300, "y": 60, "mode": "remove"})
    render(client, pid)
    assert rn.n == 2, "tap correction must change the key"
    sku = next(s for s in client.get("/api/catalog").json() if s["id"] == "WL-GRG-01")
    client.put("/api/catalog/WL-GRG-01", json={"price": sku["price"] + 1})
    render(client, pid)
    assert rn.n == 3, "price change must change the key (record carries the BOM)"
    client.post("/api/catalog/WL-GRG-01/albedo", files={"albedo": ("t.png", png_bytes(photo(128, 128, seed=99)))})
    render(client, pid)
    assert rn.n == 4, "new texture pixels must change the key"


def test_reimporting_unchanged_catalogue_keeps_cache(client, counted):
    _, rn = counted
    p = new_project(client, seed=24)
    render(client, p["id"])
    exp = client.get("/api/catalog/export").content
    done = client.post("/api/catalog/import", data={"mode": "replace", "dry_run": "false"},
                       files={"archive": ("c.zip", exp, "application/zip")}).json()
    assert done["applied"]
    render(client, p["id"])
    assert rn.n == 1


def test_other_project_reuses_render_and_survives_deletion(client, counted):
    _, rn = counted
    a = new_project(client, seed=25)
    b = new_project(client, seed=25)
    ra = render(client, a["id"])
    rb = render(client, b["id"])
    assert rn.n == 1 and rb["cached"] and rb["id"] != ra["id"] and rb["total"] == ra["total"]
    client.delete(f"/api/projects/{a['id']}")
    assert client.get(f"/api/projects/{b['id']}/renders/{rb['id']}.jpg").status_code == 200


def test_harmonise_fallback_is_never_cached(client, counted):
    _, rn = counted
    p = new_project(client, seed=26)
    render(client, p["id"], harmonize=True)
    render(client, p["id"], harmonize=True)
    assert rn.n == 2


def test_cache_is_disposable_and_capped(client, counted, monkeypatch):
    _, rn = counted
    srv = client.server
    p = new_project(client, seed=27)
    render(client, p["id"])
    assert client.delete("/api/cache").status_code == 200
    render(client, p["id"], edges=False)                    # works with an empty cache
    assert rn.n == 2
    monkeypatch.setattr(srv.cache, "limit", 1)              # 1 byte: everything must be pruned
    render(client, p["id"], grain=False)
    st = client.get("/api/cache").json()
    assert st["analysis"] == 0 and st["renders"] == 0
