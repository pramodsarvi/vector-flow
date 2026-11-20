"""Region identity: a photo's surfaces are addressable instances, not one mask per class.

A region id is ``"<kind>#<n>"`` — `kind` is the detector class, `n` a stable index. Analysis
numbers a class's regions largest first, so re-analysing the same photo reproduces the same ids;
a region minted later by tap-to-correct is appended above the highest index ever used (see
`high_index`), so the ids of a corrected scene are no longer in area order. Everything that used
to key on a bare class name now keys on a region id; code that needs the class calls `kind_of`,
code that needs every region of a class calls `by_kind`.

A bare class name is still a valid key everywhere a caller supplies one (a stored project from
before regions existed, an `assign` written by hand, the demo API examples): `kind_of` passes it
through unchanged, and `expand_assign` fans it out over the regions that exist.
"""
from __future__ import annotations

RID_SEP = "#"

# masks that are not regions: tracked so they can occlude, never assigned a material
NON_REGION = ("occluder",)

# how a class's pixels are divided into regions (applied by Engine._instance)
SINGLE = "single"        # one region, whatever the geometry says
COMPONENT = "component"  # one per connected component of the class mask
PLANE = "plane"          # one per fitted plane
DETECTION = "detection"  # one per detector box group

INSTANCING = {
    "floor": SINGLE,          # one floor: an island splitting it in two is still one floor
    "countertop": COMPONENT,  # coplanar but separate runs — island vs back counter
    "backsplash": COMPONENT,
    "wall": PLANE,            # per wall, and a fridge occluding the middle doesn't split it
    "ceiling": SINGLE,        # one ceiling, like the floor: a beam across it doesn't make two
    "cabinet": DETECTION,     # per door
}
DEFAULT_INSTANCING = COMPONENT


def instancing(kind: str) -> str:
    return INSTANCING.get(kind_of(kind), DEFAULT_INSTANCING)


def make_rid(kind: str, n: int) -> str:
    return f"{kind}{RID_SEP}{n}"


def kind_of(key: str) -> str:
    """The class of a region id. Bare class names and non-region keys pass through unchanged."""
    return key.split(RID_SEP, 1)[0]


def is_region(key: str) -> bool:
    return kind_of(key) not in NON_REGION


def regions(mapping: dict) -> dict:
    """Only the region entries of a mask/surface dict (drops `occluder` and friends)."""
    return {k: v for k, v in mapping.items() if is_region(k)}


def by_kind(mapping: dict, *kinds: str) -> dict:
    """``{rid: value}`` for every region of these classes, in the mapping's own order."""
    want = set(kinds)
    return {k: v for k, v in mapping.items() if kind_of(k) in want}


def high_index(keys, kind: str) -> int:
    """Highest region index present for `kind`, or -1 if it has none.

    Callers mint `high_index(...) + 1` and never fill a gap: an index freed by a region that was
    corrected away must not be handed out again. `meta["assign"]` (and so the UI's material
    choices) outlives a correction, and reusing an index would silently re-bind a stored
    assignment to a different physical surface.
    """
    high = -1
    for k in keys:
        if kind_of(k) != kind:
            continue
        _, _, tail = k.partition(RID_SEP)
        if tail.isdigit():
            high = max(high, int(tail))
    return high


def union_of_kind(masks: dict, *kinds: str):
    """Every region of these classes OR'd together, or None if there are none.

    For the pipeline's class-level questions ("is there enough floor to refine gravity?"), which
    care about the class as a whole rather than any one region.
    """
    ms = list(by_kind(masks, *kinds).values())
    if not ms:
        return None
    out = ms[0].copy()
    for m in ms[1:]:
        out |= m
    return out


def largest_of_kind(surfs: dict, *kinds: str):
    """The planar region of the first of `kinds` that has any — biggest mask wins.

    For the "pick one reference surface" lookups (`scale_from_prior`, `align_horizontal_axes`),
    which want a single representative plane, not every region of the class.
    """
    for k in kinds:
        cands = [s for s in by_kind(surfs, k).values() if getattr(s, "planes", None)]
        if cands:
            return max(cands, key=lambda s: int(s.mask.sum()))
    return None


def expand_assign(assign: dict, keys) -> dict:
    """Resolve an `assign` written with class names into one keyed by region id.

    An explicit region id is kept as-is; a bare class name is applied to every region of that
    class; a key matching nothing is kept untouched, so the caller still sees it (the renderer
    skips keys with no surface, exactly as it did before regions existed).
    """
    keys = list(keys)
    out = {}
    for key, value in assign.items():
        if key in keys:
            out[key] = value
            continue
        matched = [k for k in keys if kind_of(k) == key]
        for k in matched:
            out[k] = value
        if not matched:
            out[key] = value
    return out
