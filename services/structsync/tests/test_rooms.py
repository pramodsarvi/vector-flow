"""Any indoor space, not just kitchens: ceilings, and the metric scale prior that replaces the
counter-height assumption when there is no counter in the photo.

The two things this pins are the ones that fail *quietly*. A ceiling and a floor are the two
largest horizontal planes in a room, and the orientation filter used to test `|n·up|`, which
cannot tell them apart. And a room with no countertop used to get no scale prior at all, so its
areas simply drifted — a wrong number, not an error.
"""
import pathlib

import numpy as np
import pytest
from conftest import ROOT, photo, two_wall_room

import structsync.catalog as C
import structsync.pipeline as P
import structsync.vision as V
from structsync.regions import SINGLE, instancing, kind_of
from structsync.surfaces import (
    CEILING_HEIGHT_M,
    COUNTER_HEIGHT_M,
    DOWN_FACING,
    HEIGHT_PRIORS,
    UP_FACING,
    Plane,
    Surface,
    _plane_gap,
    orientation_filter,
)


@pytest.fixture
def engine(monkeypatch):
    monkeypatch.setattr(C, "ROOT", ROOT / "tests" / "fixtures" / "catalog")
    return P.Engine()


def analysed(engine, **kw):
    state = two_wall_room(480, 360, **kw)
    sc = engine.restore(photo(480, 360), state, [], {})
    engine._instance(sc)
    return sc, state


def scale_note(sc):
    return next(n for n in sc.notes if n.startswith(("scaled", "metric", "prior")))


# ---------------------------------------------------------------- signed orientation
def test_a_floor_is_not_accepted_as_a_ceiling_or_the_reverse(engine):
    """`orientation_filter` tested `|n·up| > 0.8`, which accepts both facings. Measured against
    `point_normals`, an up-facing surface is n·up ≈ −1 and a down-facing one ≈ +1, so without the
    sign a ceiling passes as a floor and a floor passes as a ceiling."""
    state = two_wall_room(480, 360)
    sc = engine.restore(photo(480, 360), state, [], {})
    floor, ceil = state["mask_floor"], state["mask_ceiling"]
    assert floor.sum() > 1000 and ceil.sum() > 1000, "fixture must show both"

    def kept(kind, mask):
        return orientation_filter(kind, mask, sc.normals, sc.up).sum() / mask.sum()

    assert kept("floor", floor) > 0.9 and kept("ceiling", ceil) > 0.9
    # the two are mutually exclusive to measurement precision: this is what |n·up| could not do
    assert kept("ceiling", floor) < 0.001, "a floor must not satisfy the ceiling criterion"
    assert kept("floor", ceil) < 0.001, "a ceiling must not satisfy the floor criterion"
    # a wall satisfies neither, bar a thin band at each seam: `point_normals` blurs the point map
    # (sigma 2 px), so normals a pixel or two from the wall/ceiling junction do tilt over
    assert kept("floor", state["mask_wall"]) < 0.05
    assert kept("ceiling", state["mask_wall"]) < 0.05


def test_the_facing_tables_stay_disjoint_and_cover_the_horizontal_kinds():
    assert not set(UP_FACING) & set(DOWN_FACING)
    assert "ceiling" in DOWN_FACING and "floor" in UP_FACING and "countertop" in UP_FACING


# ---------------------------------------------------------------- ceiling as a surface
def test_ceiling_is_detected_as_one_horizontal_region(engine):
    sc, _ = analysed(engine)
    assert [r for r in sc.surfaces if kind_of(r) == "ceiling"] == ["ceiling#0"]
    s = sc.surfaces["ceiling#0"]
    assert len(s.planes) == 1 and s.planes[0].horizontal
    assert s.area_m2 > 1.0
    assert s.kind == "ceiling"


def test_a_ceiling_never_splits():
    """Like the floor: a beam or a light across it doesn't make it two ceilings."""
    assert instancing("ceiling") == SINGLE
    assert instancing("ceiling#0") == SINGLE


def test_the_ceiling_plane_is_distinct_from_the_floor_plane(engine):
    sc, _ = analysed(engine)
    cn = sc.surfaces["ceiling#0"].planes[0].n
    fn = sc.surfaces["floor#0"].planes[0].n
    assert float(cn @ sc.up) > 0.8 and float(fn @ sc.up) < -0.8


def test_a_ceiling_takes_a_material_of_its_own(engine):
    sc, _ = analysed(engine)
    sku = next(s for s in engine.catalog if s.category == "ceiling")
    _, bom, _ = engine.render(sc, {"ceiling": sku.id}, check_sku=False)
    assert [b["surface"] for b in bom] == ["ceiling"]
    assert bom[0]["regions"] == ["ceiling#0"] and bom[0]["area_m2"] > 1.0


def test_tap_to_correct_works_on_a_ceiling(engine):
    sc, state = analysed(engine)

    class Click:
        def mask_from_point(self, img, xy, positive=True):
            w, h = img.size
            yy, xx = np.mgrid[0:h, 0:w]
            return (xx - xy[0]) ** 2 + (yy - xy[1]) ** 2 < 30 ** 2

    engine._seg = Click()
    before = int(sc.masks["ceiling#0"].sum())
    ys, xs = np.nonzero(state["mask_ceiling"])
    i = len(xs) // 2
    at = (int(xs[i]), int(ys[i]))
    engine.edit_mask(sc, "ceiling", at, add=False, region="ceiling#0")
    assert sc.masks["ceiling#0"].sum() < before

    # `add` is the path that runs the signed orientation filter, and a sign error there drops
    # every added pixel silently. `remove` is a plain mask subtraction and would never show it.
    sc.masks["ceiling#0"] = np.zeros_like(sc.masks["ceiling#0"])
    engine.refit(sc)
    assert "ceiling#0" not in sc.masks
    engine.edit_mask(sc, "ceiling", at, add=True)
    grown = [r for r in sc.masks if kind_of(r) == "ceiling"]
    assert grown, "an added ceiling region was filtered away entirely"
    assert int(sc.masks[grown[0]].sum()) > 100
    assert sc.surfaces[grown[0]].planes[0].horizontal


# ---------------------------------------------------------------- the scale prior
def test_a_kitchen_still_uses_the_counter_height(engine):
    sc, _ = analysed(engine)
    note = scale_note(sc)
    assert note.startswith("scaled") and "countertop over floor" in note
    assert sc.scale == pytest.approx(1.0, abs=0.05)


def test_a_room_with_no_counter_is_scaled_from_the_ceiling(engine):
    """This is the whole point of Phase 1's prior work: a bedroom, hallway or bathroom has no
    countertop, and before this it got no prior at all and its areas drifted."""
    sc, _ = analysed(engine, counter=False)
    note = scale_note(sc)
    assert note.startswith("scaled"), note
    assert "ceiling over floor" in note and "countertop" not in note
    assert sc.scale == pytest.approx(1.0, abs=0.05)
    assert sc.surfaces["floor#0"].area_m2 > 1.0


def test_a_counter_outvotes_a_ceiling_when_they_disagree(engine):
    """A counter is a standard 0.91 m almost everywhere; ceilings vary by half a metre. So when
    both pairs are visible and disagree, the answer must sit near the counter's."""
    sc, _ = analysed(engine, ceiling_h=1.8)          # 3.0 m floor-to-ceiling, counter still 0.91
    note = scale_note(sc)
    assert "countertop over floor" in note and "ceiling over floor" in note
    ceiling_only = CEILING_HEIGHT_M / 3.0            # what the ceiling alone would have said
    assert abs(sc.scale - 1.0) < abs(sc.scale - ceiling_only), note
    assert sc.scale == pytest.approx(1.0, abs=0.1)


def test_an_implausible_measurement_is_rejected_not_applied(engine):
    """A 1.7 m floor-to-ceiling is a mis-segmentation, not a room. Scaling by it would be worse
    than leaving the depth alone."""
    sc, _ = analysed(engine, counter=False, ceiling_h=0.5)
    note = scale_note(sc)
    assert note.startswith("prior rejected"), note
    assert "1.70" in note
    assert sc.scale == 1.0


def test_no_known_pair_falls_back_to_raw_depth(engine):
    """A photo of a wall and nothing else: there is no prior to apply, and the note has to say so
    rather than silently claim a scale."""
    state = two_wall_room(480, 360, counter=False)
    state["mask_floor"] = np.zeros_like(state["mask_floor"])
    sc = engine.restore(photo(480, 360), state, [], {})
    note = scale_note(sc)
    assert note.startswith("metric depth only"), note
    assert sc.scale == 1.0


def test_every_prior_is_a_plausible_range_around_its_expected_height():
    for upper, lower, expect, (lo, hi), weight in HEIGHT_PRIORS:
        assert lo < expect < hi, (upper, lower)
        assert 0 < weight <= 1.0
    kinds = {p[0] for p in HEIGHT_PRIORS} | {p[1] for p in HEIGHT_PRIORS}
    assert kinds <= set(V.SURFACES), kinds - set(V.SURFACES)
    assert COUNTER_HEIGHT_M < CEILING_HEIGHT_M


def test_the_counter_prior_is_weighted_above_the_ceiling_prior():
    w = {p[0]: p[4] for p in HEIGHT_PRIORS}
    assert w["countertop"] > w["ceiling"]


# ---------------------------------------------------------------- prompt vocabulary
def test_the_kitchen_surface_prompt_is_unchanged_by_the_new_classes():
    """The README records that putting the whole-run cabinet phrases into the surface prompt made
    DINO drop whole walls and most of the backsplash. Ceiling is therefore isolated in its own
    pass rather than appended to a prompt that is known to work."""
    passes = V.prompt_passes()
    surf = passes[0]
    assert surf == ["countertop", "kitchen counter", "cabinet", "cabinet door", "drawer",
                    "backsplash tile", "backsplash", "floor", "wall"]
    assert "ceiling" not in surf
    assert list(V.CEILING_PHRASES) in passes
    assert list(V.RUN_PHRASES) in passes


def test_every_prompt_phrase_is_detected_in_exactly_one_pass():
    flat = [p for chunk in V.prompt_passes() for p in chunk]
    assert sorted(flat) == sorted(V.PROMPTS), "a phrase was dropped or duplicated"
    assert all(len(c) <= V.MAX_PHRASES_PER_PASS for c in V.prompt_passes())
    assert all(c for c in V.prompt_passes()), "no empty pass"


def test_non_kitchen_furniture_is_occluded_rather_than_painted_over():
    """The occluder vocabulary, not the surface vocabulary, is what decides whether a bedroom or
    bathroom photo comes out usable: a bed left off this list gets the wall's new paint projected
    straight across it."""
    for thing in ("bed", "sofa", "couch", "wardrobe", "curtain", "mirror", "rug",
                  "toilet", "bathtub", "television", "ceiling fan", "door"):
        assert V.PROMPTS.get(thing) == "occluder", thing


def test_priority_covers_every_surface_class_and_ranks_ceiling_last():
    assert set(V.PRIORITY) == set(V.SURFACES) | {"occluder"}
    assert V.PRIORITY["occluder"] == max(V.PRIORITY.values())
    # a ceiling is only ever behind other things, so it must never win a contested pixel
    assert V.PRIORITY["ceiling"] == min(V.PRIORITY.values())


# ---------------------------------------------------------------- catalogue
def test_the_catalogue_offers_ceiling_materials(engine):
    ceilings = [s for s in engine.catalog if s.category == "ceiling"]
    assert len(ceilings) >= 3, [s.id for s in ceilings]
    for s in ceilings:
        assert (s.path / "albedo.png").exists()
        assert s.tile_w > 0 and s.tile_h > 0 and 0 <= s.gloss <= 1
    # ceiling paint must tile large enough not to show a repeat across a whole ceiling
    assert max(s.tile_w for s in ceilings) >= 2.0


def test_every_surface_class_has_at_least_one_material(engine):
    """Otherwise the UI offers a surface it cannot restyle."""
    have = {s.category for s in engine.catalog}
    assert set(V.SURFACES) <= have, set(V.SURFACES) - have


def test_the_committed_catalogue_is_what_the_generator_produces():
    """Guards the append: the demo albedos are seeded by list index, so inserting a SKU anywhere
    above the end would silently regenerate every texture below it."""
    import json
    fixture = json.loads((ROOT / "tests" / "fixtures" / "catalog" / "catalog.json").read_text())
    assert [e["id"] for e in fixture[:12]] == [d[0] for d in C.DEMO[:12]]
    assert [e["id"] for e in fixture] == [d[0] for d in C.DEMO]


def test_the_shipped_catalogue_carries_the_ceiling_skus():
    import json
    ship = pathlib.Path(__file__).resolve().parent.parent / "catalog"
    entries = json.loads((ship / "catalog.json").read_text())
    ceilings = [e for e in entries if e["category"] == "ceiling"]
    assert len(ceilings) >= 3, [e["id"] for e in ceilings]
    for e in ceilings:
        assert (ship / e["id"] / "albedo.png").exists(), e["id"]


def test_a_height_prior_refuses_a_plane_that_faces_the_wrong_way():
    """`Plane.horizontal` is unsigned, so it alone lets an up-facing plane stand in for a ceiling.
    `_plane_gap` is the second consumer of the facing distinction after `orientation_filter`, and
    it has to apply the same signed test — otherwise a tabletop mistaken for a ceiling rescales
    the whole scene from a height that looks entirely plausible."""
    up = np.array([0.0, -1.0, 0.0])
    m = np.ones((8, 8), bool)

    def surf(kind, n, d):
        pl = Plane(np.array(n, float), float(d), np.array([1.0, 0, 0]), np.array([0.0, 0, 1]),
                   m, True)
        return Surface(kind, m, [pl], rid=kind + "#0")

    floor = surf("floor", (0, 1, 0), 1.2)        # n·up = −1: up-facing, 1.2 m below the camera
    ceil = surf("ceiling", (0, -1, 0), 1.3)      # n·up = +1: down-facing, 1.3 m above it
    good = {"floor#0": floor, "ceiling#0": ceil}
    assert _plane_gap(good, "ceiling", "floor", up) == pytest.approx(2.5)

    # same height, but fitted to an up-facing plane: refused, though 2.5 m is perfectly plausible
    bogus = {"floor#0": floor, "ceiling#0": surf("ceiling", (0, 1, 0), -1.3)}
    assert _plane_gap(bogus, "ceiling", "floor", up) is None

    # and a pair measured the wrong way round is refused rather than scaled by
    assert _plane_gap(good, "floor", "ceiling", up) is None


def test_a_vertical_surface_never_supplies_a_height():
    up = np.array([0.0, -1.0, 0.0])
    m = np.ones((8, 8), bool)
    wall = Plane(np.array([0.0, 0, 1]), 3.0, np.array([1.0, 0, 0]), np.array([0.0, 1, 0]), m, False)
    floor_p = Plane(np.array([0.0, 1, 0]), 1.2, np.array([1.0, 0, 0]), np.array([0.0, 0, 1]), m, True)
    surfs = {"wall#0": Surface("wall", m, [wall], rid="wall#0"),
             "floor#0": Surface("floor", m, [floor_p], rid="floor#0")}
    assert _plane_gap(surfs, "wall", "floor", up) is None
