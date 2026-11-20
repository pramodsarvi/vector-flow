"""Region identity: the pipeline addresses surfaces as instances, not one mask per class.

These tests exist mostly to catch *silent* regressions. Keying surfaces by `"wall#0"` instead of
`"wall"` breaks exact-match class lookups, and almost every one of those lookups fails by
returning None and skipping its step — a lost scale prior or lost axis alignment still renders a
plausible-looking image. The suite below pins the behaviour of each of them.
"""
import numpy as np
import pytest
from conftest import ROOT, photo, room_state, two_wall_room
from PIL import Image

import structsync.catalog as C
import structsync.pipeline as P
import structsync.render as R
import structsync.vision as V
from structsync.regions import expand_assign, high_index, kind_of, make_rid
from structsync.surfaces import Surface, fit_surface
from structsync.viz import region_color


@pytest.fixture
def engine(monkeypatch):
    """A real Engine over the fixture catalogue. Models stay unloaded (they are lazy, and
    conftest's autouse fixture makes every constructor raise)."""
    monkeypatch.setattr(C, "ROOT", ROOT / "tests" / "fixtures" / "catalog")
    return P.Engine()


def analysed(engine, state, img=None):
    """A scene taken through the analysis-time split, the way `analyze` does it."""
    img = img or photo(480, 360)
    sc = engine.restore(img, state, [], {})
    engine._instance(sc)
    return sc


# ---------------------------------------------------------------- ids
def test_kind_of_handles_regions_bare_classes_and_derived_edges():
    assert kind_of("wall#0") == "wall" and kind_of("wall") == "wall"
    assert kind_of("occluder") == "occluder"
    assert kind_of(make_rid("cabinet", 12)) == "cabinet"


def test_high_index_reports_the_top_index_per_class():
    assert high_index(["wall#0", "wall#1", "floor#0"], "wall") == 1
    assert high_index(["wall#1"], "wall") == 1
    assert high_index([], "cabinet") == -1
    assert high_index(["wall"], "wall") == -1             # a bare class name holds no index


def test_a_freed_region_index_is_never_handed_out_again(engine):
    """`meta["assign"]` outlives a correction and the UI repopulates its material choices from it.
    If an index freed by an emptied region were reused, that stored assignment would silently
    re-bind to a different physical surface."""
    sc = analysed(engine, two_wall_room(480, 360))
    engine._seg = _Click()
    engine.edit_mask(sc, "cabinet", (240, 180), add=True)
    assert "cabinet#0" in sc.masks
    sc.masks["cabinet#0"] = np.zeros_like(sc.masks["cabinet#0"])
    engine.refit(sc)
    assert "cabinet#0" not in sc.masks                    # corrected away entirely
    engine.edit_mask(sc, "cabinet", (120, 300), add=True)
    assert "cabinet#0" not in sc.masks, "a freed index must not be reused"
    assert "cabinet#1" in sc.masks


def test_the_monotonic_index_survives_a_restart(engine):
    sc = analysed(engine, two_wall_room(480, 360))
    engine._seg = _Click()
    engine.edit_mask(sc, "cabinet", (240, 180), add=True)   # cabinet#0
    sc.masks["cabinet#0"] = np.zeros_like(sc.masks["cabinet#0"])
    engine.refit(sc)
    back = engine.restore(photo(480, 360), engine.scene_state(sc), [], {})
    engine.edit_mask(back, "cabinet", (120, 300), add=True)
    assert "cabinet#1" in back.masks and "cabinet#0" not in back.masks


def test_expand_assign_fans_a_class_over_its_regions():
    keys = ["wall#0", "wall#1", "floor#0"]
    assert expand_assign({"wall": "A"}, keys) == {"wall#0": "A", "wall#1": "A"}
    # an explicit region wins over nothing else; the other wall is simply not assigned
    assert expand_assign({"wall#1": "B"}, keys) == {"wall#1": "B"}
    assert expand_assign({"wall": "A", "wall#1": "B"}, keys)["wall#1"] == "B"
    # a key matching no region survives, so the renderer can skip it exactly as it always has
    assert expand_assign({"ceiling": "C"}, keys) == {"ceiling": "C"}


def test_region_colours_separate_neighbours_of_one_class():
    assert region_color("wall#0") != region_color("wall#1")
    assert region_color("wall#0")[:3] != region_color("floor#0")[:3]


# ---------------------------------------------------------------- instancing
def test_wall_splits_by_plane_and_floor_stays_single(engine):
    sc = analysed(engine, two_wall_room(480, 360))
    walls = [r for r in sc.surfaces if kind_of(r) == "wall"]
    assert sorted(walls) == ["wall#0", "wall#1"], sorted(sc.surfaces)
    assert [r for r in sc.surfaces if kind_of(r) == "floor"] == ["floor#0"]
    # the two wall regions are genuinely different planes, not a split of one
    n0, n1 = (sc.surfaces[w].planes[0].n for w in ("wall#0", "wall#1"))
    assert abs(float(n0 @ n1)) < 0.9


def test_region_ids_are_largest_first_and_cover_the_class_without_loss(engine):
    state = two_wall_room(480, 360)
    sc = analysed(engine, state)
    walls = [sc.masks[r] for r in ("wall#0", "wall#1")]
    assert walls[0].sum() >= walls[1].sum()                      # largest first
    assert not (walls[0] & walls[1]).any()                       # one owner per pixel
    # no pixel of the class is dropped by the split
    assert np.array_equal(walls[0] | walls[1], state["mask_wall"])


def test_splitting_is_deterministic_so_the_cache_key_is_stable(engine):
    """test_server pins `scene_rev` equality across a re-analyse, and scene_rev hashes an npz whose
    keys now carry region indices. Component labels and group iteration are both order-sensitive."""
    state, img = two_wall_room(480, 360), photo(480, 360)
    a, b = analysed(engine, state, img), analysed(engine, state, img)
    assert sorted(a.masks) == sorted(b.masks)
    assert all(np.array_equal(a.masks[k], b.masks[k]) for k in a.masks)


def test_occluder_is_never_a_region(engine):
    state = two_wall_room(480, 360)
    state["mask_occluder"] = np.zeros_like(state["mask_floor"])
    state["mask_occluder"][:40, :40] = True
    sc = analysed(engine, state)
    assert "occluder" in sc.masks and "occluder#0" not in sc.masks
    assert "occluder" not in sc.surfaces           # never fitted, never assigned a material


# ---------------------------------------------------------------- the silent class lookups
def test_counter_height_scale_prior_still_fires_with_region_keys(engine):
    """`scale_from_prior` used `surfs.get("countertop")`. Against region keys that returns None and
    the function falls back to raw metric depth — silently, with every BOM area wrong."""
    sc = analysed(engine, two_wall_room(480, 360))
    note = next(n for n in sc.notes if "scale" in n or "metric" in n or "prior" in n)
    assert "no floor+counter pair" not in note, note
    assert note.startswith("scaled"), note
    assert sc.scale == pytest.approx(1.0, abs=0.05)   # the room is built at the real counter height


def test_horizontal_axes_align_to_a_wall_through_region_keys(engine):
    """`align_horizontal_axes` used `surfs.get("wall")`; with region keys `ref` stays None and
    counter veins/planks quietly follow the camera instead of the room. The room is built with an
    oblique back wall on purpose: against axis-aligned walls a horizontal plane's default texture
    axis (camera x) is already perpendicular to the wall, and the test would pass either way."""
    sc = analysed(engine, two_wall_room(480, 360, yaw=np.radians(25)))
    counter = sc.surfaces["countertop#0"].planes[0]
    walls = [sc.surfaces[r].planes[0].n for r in sc.surfaces if kind_of(r) == "wall"]
    # no wall is perpendicular to camera x, so "aligned" cannot happen by accident here
    assert all(abs(float(np.array([1.0, 0, 0]) @ n)) > 0.3 for n in walls)
    assert abs(float(counter.eu @ counter.n)) < 1e-6              # u lies in the counter plane
    assert min(abs(float(counter.eu @ n)) for n in walls) < 1e-6  # and runs along a wall


def test_gravity_is_recovered_from_the_floor_class(engine):
    """`refit`'s gravity loop asked `sc.masks.get("floor")`. Against region keys that is None, the
    refinement is skipped and `up` silently keeps its initial guess — which shifts every
    orientation test, every measured height and therefore every area."""
    pitch = np.radians(15)
    state = two_wall_room(480, 360, pitch=pitch)
    cp, sp = np.cos(pitch), np.sin(pitch)
    true_up = -np.array([[1.0, 0, 0], [0, cp, -sp], [0, sp, cp]]) @ np.array([0.0, 1.0, 0.0])
    assert float(state["up"] @ true_up) < 0.99, "the stored guess must be genuinely wrong"
    sc = engine.restore(photo(480, 360), state, [], {})
    assert float(sc.up @ true_up) > 0.999, (sc.up, true_up)


def test_floor_region_is_fitted_as_horizontal(engine):
    """`orientation_filter` tested `kind in HORIZONTAL_KINDS`; with a region id the floor would be
    held to the *vertical* criterion and its mask destroyed."""
    sc = analysed(engine, two_wall_room(480, 360))
    floor = sc.surfaces["floor#0"]
    assert floor.planes and floor.planes[0].horizontal
    assert floor.area_m2 > 1.0


def test_fit_surface_forces_one_plane_for_a_counter_region(engine):
    """`fit_surface` tested `kind in ("countertop","floor")` to force a single plane — counters and
    floors are flat by definition, so a second plane is a mis-segmentation. A region id has to hit
    that rule exactly as the bare class did. Tested here rather than through `_instance`, which
    would split the two planes into separate regions and hide the difference."""
    state = two_wall_room(480, 360)
    sc = engine.restore(photo(480, 360), state, [], {})
    # one mask spanning two horizontal planes: the counter band plus a slab of floor
    both = state["mask_countertop"] | state["mask_floor"]
    assert len(fit_surface("backsplash", both, sc.points, sc.valid, sc.up).planes) == 2, \
        "fixture must span two planes, or this test proves nothing"
    for key in ("countertop", "countertop#0", "floor#3"):
        s = fit_surface(key, both, sc.points, sc.valid, sc.up)
        assert len(s.planes) == 1, (key, len(s.planes))
        assert s.kind == kind_of(key) and s.rid == key


def test_surface_records_its_class_and_its_region_separately(engine):
    sc = analysed(engine, two_wall_room(480, 360))
    s = sc.surfaces["wall#1"]
    assert s.kind == "wall" and s.rid == "wall#1"


def test_orient_keeps_the_floor_once_regions_are_keyed(engine):
    """`_orient` runs `orientation_filter` per region. The filter tested `kind in HORIZONTAL_KINDS`,
    so a region id made it hold the floor to the *vertical* criterion and wipe the mask. Only
    `analyze` calls `_orient`, so nothing else in the suite exercises it."""
    state = two_wall_room(480, 360)
    sc = engine.restore(photo(480, 360), state, [], {})
    floor_px = int(sc.masks["floor#0"].sum())
    engine._orient(sc)
    assert "floor#0" in sc.surfaces, "the floor was filtered away as if it were a wall"
    assert sc.masks["floor#0"].sum() > 0.5 * floor_px
    assert sc.surfaces["floor#0"].planes[0].horizontal


def _two_counter_state(w=480, h=360):
    """`two_wall_room` with the counter band split in two, so the counter class has two regions."""
    state = two_wall_room(w, h)
    gap = np.zeros((h, w), bool)
    gap[:, int(0.45 * w):int(0.55 * w)] = True
    counter = state["mask_countertop"]            # capture before masking the gap out
    state["mask_countertop"] = counter & ~gap
    state["mask_wall"] = state["mask_wall"] | (counter & gap)
    return state


def _fake_front_edge(surface, *a, **k):
    """A 6-row band directly under the region, which is what `front_edge` would find on a photo
    (its own geometry checks reject the synthetic band: there is no real slab face below it)."""
    ys, xs = np.nonzero(surface.mask)
    band = np.zeros_like(surface.mask)
    y0 = int(ys.max()) + 1
    band[y0:y0 + 6, int(xs.min()):int(xs.max()) + 1] = True
    band &= ~surface.mask
    e = Surface(surface.kind + "_edge", band, list(surface.planes[:1]),
                rid=surface.rid + "_edge")
    e.uv = np.zeros(band.shape + (2,), np.float32)
    e.footprint = (band * 1e-3).astype(np.float32)
    return e


def test_slab_edges_are_keyed_by_region_not_class(engine, monkeypatch):
    sc = analysed(engine, _two_counter_state())
    assert sorted(r for r in sc.surfaces if kind_of(r) == "countertop") == \
        ["countertop#0", "countertop#1"], sorted(sc.surfaces)
    monkeypatch.setattr(P, "front_edge", _fake_front_edge)
    engine.refit(sc)
    assert sorted(sc.edges) == ["countertop#0", "countertop#1"]
    assert [sc.edges[k].rid for k in sorted(sc.edges)] == \
        ["countertop#0_edge", "countertop#1_edge"]


def test_each_counter_region_renders_its_own_slab_edge(engine, monkeypatch, tmp_path):
    """Keying the edge by class (`countertop_edge`) makes two counters collide on one key, so one
    of them silently loses its edge. Only a render shows that."""
    sc = analysed(engine, _two_counter_state())
    monkeypatch.setattr(P, "front_edge", _fake_front_edge)
    engine.refit(sc)
    a, b = _sku(tmp_path, "CT-A", 7, "countertop"), _sku(tmp_path, "CT-B", 8, "countertop")
    monkeypatch.setitem(engine.by_id, "CT-A", a)
    monkeypatch.setitem(engine.by_id, "CT-B", b)
    captured = {}
    real = R.project_materials

    def spy(image, surfaces, assign, **kw):
        captured.update(assign)
        return real(image, surfaces, assign, **kw)

    monkeypatch.setattr(P, "project_materials", spy)
    engine.render(sc, {"countertop#0": "CT-A", "countertop#1": "CT-B"}, check_sku=False, edges=True)
    assert captured["countertop#0_edge"].id == "CT-A"
    assert captured["countertop#1_edge"].id == "CT-B", "the second counter lost its edge"


# ---------------------------------------------------------------- rendering per region
def _sku(tmp_path, sid, seed, category="wall"):
    d = tmp_path / sid
    d.mkdir()
    rng = np.random.default_rng(seed)
    Image.fromarray((rng.random((64, 64, 3)) * 255).astype(np.uint8)).save(d / "albedo.png")
    return C.SKU(id=sid, name=sid, category=category, vendor="v", price=10.0, tile_w=0.5,
                 tile_h=0.5, gloss=0.3, path=d)


def test_two_regions_of_one_class_take_different_materials(engine, tmp_path):
    sc = analysed(engine, two_wall_room(480, 360))
    a, b = _sku(tmp_path, "A", 1), _sku(tmp_path, "B", 2)
    surfs = {k: s for k, s in sc.surfaces.items()}
    out, edited, info = R.project_materials(sc.full_rgb, surfs, {"wall#0": a, "wall#1": b}, up=sc.up)
    assert {"wall#0", "wall#1"} <= set(info)
    m0, m1 = info["wall#0"]["mask"], info["wall#1"]["mask"]
    assert m0.any() and m1.any() and not (m0 & m1).any()
    # the two walls really did get different textures, not one material twice
    only_a, _, _ = R.project_materials(sc.full_rgb, surfs, {"wall#0": a, "wall#1": a}, up=sc.up)
    assert np.abs(out[m1] - only_a[m1]).max() > 1e-3
    assert np.abs(out[m0] - only_a[m0]).max() < 1e-9


def test_render_bom_aggregates_regions_sharing_a_material(engine, tmp_path, monkeypatch):
    sc = analysed(engine, two_wall_room(480, 360))
    sku = _sku(tmp_path, "WL-X", 3)
    monkeypatch.setitem(engine.by_id, "WL-X", sku)
    _, bom, _ = engine.render(sc, {"wall": "WL-X"}, check_sku=False)
    assert len(bom) == 1, bom
    line = bom[0]
    assert line["surface"] == "wall" and sorted(line["regions"]) == ["wall#0", "wall#1"]
    total = sum(sc.surfaces[r].area_m2 for r in ("wall#0", "wall#1"))
    assert line["area_m2"] == pytest.approx(round(total, 2))
    assert line["est_cost"] == round(total * sku.price)


def test_render_accepts_a_single_region(engine, tmp_path, monkeypatch):
    sc = analysed(engine, two_wall_room(480, 360))
    sku = _sku(tmp_path, "WL-X", 3)
    monkeypatch.setitem(engine.by_id, "WL-X", sku)
    _, bom, _ = engine.render(sc, {"wall#1": "WL-X"}, check_sku=False)
    assert [b["regions"] for b in bom] == [["wall#1"]]


def test_sku_recognition_ranks_against_the_class_not_the_region_id():
    """`rank_image` filters the catalogue by category. Handed "wall#0" it matches nothing and
    reports rank=None of 0 — a verification pass that silently stops verifying."""
    asked = []

    class Search:
        def rank_image(self, img, category=None):
            asked.append(category)
            return [(C.SKU(id="S", name="s", category="wall", vendor="v", price=1, tile_w=1,
                           tile_h=1, gloss=0, path=None), 1.0)]

    m = np.zeros((20, 20), bool)
    m[5:15, 5:15] = True
    skus = {"wall#3": C.SKU(id="S", name="s", category="wall", vendor="v", price=1, tile_w=1,
                            tile_h=1, gloss=0, path=None)}
    rep = R.sku_recognition(Search(), np.zeros((20, 20, 3), np.float32), {"wall#3": dict(mask=m)}, skus)
    assert asked == ["wall"]
    assert rep["wall#3"]["pass_"] is True


# ---------------------------------------------------------------- persistence
def test_a_project_saved_before_regions_restores_as_region_zero(engine):
    """Stored scenes hold one mask per class. They must come back as that class's region 0 so the
    assignments saved against them still resolve."""
    state = room_state(480, 360)
    assert all(k.startswith("mask_") and "#" not in k for k in state if k.startswith("mask_"))
    sc = engine.restore(photo(480, 360), state, [], {})
    assert "floor#0" in sc.surfaces and "wall#0" in sc.surfaces
    assert "floor" not in sc.surfaces
    assert "occluder" in sc.masks            # still not a region
    # restore must not split: ids stay exactly as they were stored
    assert sorted(k for k in sc.masks if k != "occluder") == ["cabinet#0", "floor#0", "wall#0"]


def test_region_ids_and_corrections_survive_a_save_restore_round_trip(engine):
    sc = analysed(engine, two_wall_room(480, 360))
    state = engine.scene_state(sc)
    assert "mask_wall#0" in state and "mask_wall#1" in state
    back = engine.restore(photo(480, 360), state, [], {})
    assert sorted(back.surfaces) == sorted(sc.surfaces)
    for k in sc.surfaces:
        assert np.array_equal(back.masks[k], sc.masks[k])


def test_emptied_regions_are_dropped_rather_than_persisted(engine):
    sc = analysed(engine, two_wall_room(480, 360))
    sc.masks["wall#1"] = np.zeros_like(sc.masks["wall#1"])
    engine.refit(sc)
    assert "wall#1" not in sc.masks and "wall#1" not in sc.surfaces
    assert "wall#0" in sc.surfaces


def test_a_small_region_survives_the_class_level_noise_rule(engine):
    """`refit` dropped any wall/floor mask under 3 % of the *whole image*. Applied per region that
    silently deletes legitimate slivers; it belongs to the class as a whole."""
    sc = analysed(engine, two_wall_room(480, 360))
    big = sc.masks["wall#0"]
    small = np.zeros_like(big)
    ys, xs = np.nonzero(big)
    keep = slice(0, 400)                       # ~400 px: well under 3 % of a 480×360 photo
    small[ys[keep], xs[keep]] = True
    sc.masks["wall#0"] = big & ~small
    sc.masks["wall#9"] = small
    engine.refit(sc)
    assert "wall#9" in sc.surfaces, "a sliver of a healthy class must not be dropped"


def test_a_class_that_is_all_noise_is_still_dropped(engine):
    """The other half of the same rule: the 3 % test must still reject a whole class of crumbs."""
    state = two_wall_room(480, 360)
    crumb = np.zeros_like(state["mask_floor"])
    crumb[:16, :16] = True                     # 256 px total: past MIN_REGION_PX, under 3 %
    state["mask_floor"] = crumb
    sc = engine.restore(photo(480, 360), state, [], {})
    assert not [r for r in sc.surfaces if kind_of(r) == "floor"]


# ---------------------------------------------------------------- tap-to-correct
class _Click:
    """A SAM stand-in: a disc around the tapped pixel."""

    def mask_from_point(self, img, xy, positive=True):
        w, h = img.size
        yy, xx = np.mgrid[0:h, 0:w]
        return (xx - xy[0]) ** 2 + (yy - xy[1]) ** 2 < 30 ** 2


def test_correcting_without_a_region_mints_one(engine):
    """The "nothing was detected here at all" workflow (kitchen4/5 find no cabinet) has no region
    to grow, so an add with no region has to create one."""
    sc = analysed(engine, two_wall_room(480, 360))
    engine._seg = _Click()
    before = set(sc.masks)
    engine.edit_mask(sc, "cabinet", (240, 180), add=True)
    new = set(sc.masks) - before
    assert new == {"cabinet#0"}
    engine.edit_mask(sc, "cabinet", (120, 300), add=True)
    assert "cabinet#1" in sc.masks


def test_correcting_a_named_region_grows_only_that_one(engine):
    sc = analysed(engine, two_wall_room(480, 360))
    engine._seg = _Click()
    other = sc.masks["wall#1"].copy()
    target = sc.masks["wall#0"].sum()
    # tap inside the other wall, assigned explicitly to wall#0: wall#0 gains, wall#1 loses
    ys, xs = np.nonzero(other)
    engine.edit_mask(sc, "wall", (int(xs[len(xs) // 2]), int(ys[len(ys) // 2])), add=True,
                     region="wall#0")
    assert sc.masks["wall#0"].sum() > target
    assert sc.masks["wall#1"].sum() < other.sum()
    assert not (sc.masks["wall#0"] & sc.masks["wall#1"]).any()


def test_removing_without_a_region_clears_every_region_of_the_class(engine):
    sc = analysed(engine, two_wall_room(480, 360))
    engine._seg = _Click()
    ys, xs = np.nonzero(sc.masks["wall#0"] | sc.masks["wall#1"])
    i = len(xs) // 2
    before = sum(int(sc.masks[k].sum()) for k in ("wall#0", "wall#1"))
    engine.edit_mask(sc, "wall", (int(xs[i]), int(ys[i])), add=False)
    after = sum(int(sc.masks[k].sum()) for k in sc.masks if kind_of(k) == "wall")
    assert after < before


# ---------------------------------------------------------------- detector grouping
# `group_detections` is the only new logic the model-free suite cannot reach through the pipeline
# (nothing here loads Grounding DINO), and it is what the README's "cabinets split per detected
# door" claim rests on. It is pure box arithmetic over (kind, score, box, label) tuples.
def _det(kind, box, label=None, score=0.9):
    return (kind, score, np.array(box, float), label or kind)


def test_overlapping_boxes_on_one_door_are_one_region():
    dets = [_det("cabinet", (0, 0, 100, 100)), _det("cabinet", (5, 5, 98, 102))]
    assert V.group_detections(dets) == {"cabinet": [[0, 1]]}


def test_two_adjacent_doors_stay_separate_regions():
    """Touching but not covering each other: a run of doors must not collapse into one region."""
    dets = [_det("cabinet", (0, 0, 100, 100)), _det("cabinet", (100, 0, 200, 100))]
    assert V.group_detections(dets) == {"cabinet": [[0], [1]]}


def test_a_run_phrase_box_does_not_merge_the_doors_it_spans():
    """"upper/base cabinet" boxes are extenders that catch plain doors DINO missed one by one, and
    they span several doors at once. Letting one merge every door it covers would collapse a whole
    run into a single region and defeat per-door addressing."""
    dets = [_det("cabinet", (0, 0, 100, 100), "cabinet door"),
            _det("cabinet", (200, 0, 300, 100), "cabinet door"),
            _det("cabinet", (0, 0, 300, 100), "upper cabinet")]
    assert V.group_detections(dets) == {"cabinet": [[0], [1], [2]]}


def test_an_individual_box_always_outranks_a_run_extender():
    """So the run phrase only ever claims the doors no single box found. Confidence must not
    change that, and the class's own precedence band must not leak into the next class down."""
    door = _det("cabinet", (0, 0, 100, 100), "cabinet door", score=0.26)
    run = _det("cabinet", (0, 0, 300, 100), "upper cabinet", score=0.99)
    assert V._rank(door) > V._rank(run)
    # cross-class ordering is untouched: any cabinet still beats any wall
    wall = _det("wall", (0, 0, 400, 300), "wall", score=0.99)
    assert V._rank(run) > V._rank(wall)
    assert V.PRIORITY["cabinet"] <= V._rank(run) < V.PRIORITY["cabinet"] + 1


def test_non_detection_classes_come_back_as_one_group():
    """Walls and floors are divided by geometry later, so grouping must not pre-split them."""
    dets = [_det("wall", (0, 0, 50, 50)), _det("wall", (300, 0, 400, 80)),
            _det("floor", (0, 80, 400, 200))]
    groups = V.group_detections(dets)
    assert groups["wall"] == [[0, 1]] and groups["floor"] == [[2]]


def test_grouping_is_deterministic_and_covers_every_detection():
    dets = [_det("cabinet", (0, 0, 60, 60)), _det("cabinet", (2, 2, 61, 59)),
            _det("cabinet", (200, 0, 260, 60)), _det("wall", (0, 0, 400, 300))]
    a = V.group_detections(dets)
    assert a == V.group_detections(dets)
    seen = sorted(i for gs in a.values() for g in gs for i in g)
    assert seen == list(range(len(dets)))
