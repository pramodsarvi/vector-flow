"""Stage 1 (scene parse) and Stage 2 (geometry solve).

Grounding DINO (Apache-2.0) → boxes, SAM 2 (Apache-2.0) → masks,
MoGe-2 (MIT) → metric point map + intrinsics.
"""
from __future__ import annotations

import os

import numpy as np
import torch
from PIL import Image

from .regions import DETECTION, instancing, make_rid

SURFACES = ["countertop", "cabinet", "backsplash", "floor", "wall", "ceiling"]
# Anything a finish must not be painted over. The kitchen set came first; the rest is what any
# other indoor room puts in front of a wall or floor. A bed left out of this list gets the wall's
# new paint projected straight across it, so the occluder vocabulary — not the surface vocabulary
# — is what decides whether a bedroom or bathroom photo comes out usable.
OCCLUDERS = ["sink", "faucet", "stove", "oven", "refrigerator", "microwave", "dishwasher",
             "bottle", "plant", "bowl", "window", "range hood", "chair", "person", "light",
             # hardware stays metal: without these, handles/knobs get painted with the cabinet colour
             "cabinet handle", "knob",
             # living / bedroom
             "bed", "sofa", "couch", "armchair", "stool", "table", "desk", "wardrobe",
             "television", "bookshelf", "pillow", "cushion", "blanket", "rug",
             # fixtures and trim that read as part of the wall unless they are called out
             "curtain", "blind", "mirror", "picture frame", "door", "radiator", "staircase",
             "ceiling fan", "clock", "vase",
             # bathroom / utility
             "toilet", "bathtub", "shower", "towel", "washing machine"]
# prompt phrase -> class key
PROMPTS = {"countertop": "countertop", "kitchen counter": "countertop",
           "cabinet": "cabinet", "cabinet door": "cabinet", "drawer": "cabinet",
           # whole-run phrases catch rows of plain doors DINO misses one by one. ("kitchen island" was
           # tried and dropped: its box pulls in stools, table legs and shelving.)
           "upper cabinet": "cabinet", "base cabinet": "cabinet",
           "backsplash tile": "backsplash", "backsplash": "backsplash",
           "floor": "floor", "wall": "wall", "ceiling": "ceiling",
           **{o: "occluder" for o in OCCLUDERS}}
RUN_PHRASES = ("upper cabinet", "base cabinet")
CEILING_PHRASES = ("ceiling",)
# Phrase groups that get a DINO pass to themselves. DINO is prompt-context sensitive: putting the
# whole-run cabinet phrases in the surface prompt made it drop whole walls and most of the
# backsplash. So rather than extend the surface prompt that is known to work on kitchens, anything
# added to it later is isolated — at the cost of one more pass each.
OWN_PASS = (RUN_PHRASES, CEILING_PHRASES)
MAX_PHRASES_PER_PASS = 12      # keeps every prompt well inside DINO's ~256 text tokens
# overlap priority: higher wins a pixel. A ceiling is only ever behind things, so it sits lowest.
PRIORITY = {"occluder": 6, "countertop": 5, "backsplash": 4, "cabinet": 3, "floor": 2, "wall": 1,
            "ceiling": 0}


def pick_device():
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    return "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")


class Segmenter:
    DINO = "IDEA-Research/grounding-dino-base"
    # STRUCTSYNC_SAM=facebook/sam2.1-hiera-large to try the large model (compare masks before
    # switching the default; it must also be downloaded beforehand — nothing downloads on its own)
    SAM = os.environ.get("STRUCTSYNC_SAM", "facebook/sam2.1-hiera-small")

    def __init__(self, device):
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor, Sam2Model, Sam2Processor
        self.device = device
        self.dino_proc = AutoProcessor.from_pretrained(self.DINO)
        self.dino = AutoModelForZeroShotObjectDetection.from_pretrained(self.DINO).to(device).eval()
        self.sam_proc = Sam2Processor.from_pretrained(self.SAM)
        self.sam = Sam2Model.from_pretrained(self.SAM).to(device).eval()
        self._emb_cache = None

    @torch.no_grad()
    def detect(self, img: Image.Image, box_thr=0.25, text_thr=0.2):
        dets = []
        for chunk in prompt_passes():
            text = ". ".join(chunk) + "."
            x = self.dino_proc(images=img, text=text, return_tensors="pt").to(self.device)
            out = self.dino(**x)
            r = self.dino_proc.post_process_grounded_object_detection(
                out, x["input_ids"], threshold=box_thr, text_threshold=text_thr,
                target_sizes=[img.size[::-1]])[0]
            labels = r.get("text_labels", r.get("labels"))
            for b, s, l in zip(r["boxes"].cpu().numpy(), r["scores"].cpu().numpy(), labels):
                key = self._map(l)
                # hardware boxes must be small, otherwise SAM would cut out the whole door as an occluder
                if key and any(w in l.lower() for w in ("handle", "knob")) and \
                        _box_area(b) > 0.01 * img.size[0] * img.size[1]:
                    continue
                if key:
                    dets.append((key, float(s), b, l.lower()))
        # "upper/base cabinet" boxes only extend cabinets already seen door-by-door; on their own they
        # fire on ceiling beams, open shelving and islands-with-stools
        anchors = [d[2] for d in dets if d[0] == "cabinet" and not any(w in d[3] for w in ("upper", "base"))]
        dets = [d for d in dets if not (d[0] == "cabinet" and any(w in d[3] for w in ("upper", "base")))
                or any(_overlap(d[2], a) > 0.3 for a in anchors)]
        return dets

    @staticmethod
    def _map(label: str):
        label = label.lower().strip()
        for p in sorted(PROMPTS, key=len, reverse=True):  # longest phrase match first
            if p in label:
                return PROMPTS[p]
        return None

    @torch.no_grad()
    def masks_from_boxes(self, img, boxes):
        if len(boxes) == 0:
            return np.zeros((0, img.size[1], img.size[0]), bool)
        # encode the image once, decode boxes in chunks: one batch of 40+ boxes OOMs an 8 GB GPU
        pix = self.sam_proc(images=img, return_tensors="pt").to(self.device)
        emb = self.sam.get_image_embeddings(pix["pixel_values"])
        res = []
        for i in range(0, len(boxes), 8):
            x = self.sam_proc(images=img, input_boxes=[[list(map(float, b)) for b in boxes[i:i + 8]]],
                              return_tensors="pt").to(self.device)
            out = self.sam(input_boxes=x["input_boxes"], image_embeddings=emb, multimask_output=False)
            m = self.sam_proc.post_process_masks(out.pred_masks.cpu(), x["original_sizes"])[0]
            res.append(m[:, 0].numpy() > 0.5)
        if self.device == "cuda":
            torch.cuda.empty_cache()
        return np.concatenate(res)

    @torch.no_grad()
    def mask_from_point(self, img, xy, positive=True):
        x = self.sam_proc(images=img, input_points=[[[list(map(float, xy))]]],
                          input_labels=[[[1 if positive else 0]]], return_tensors="pt").to(self.device)
        out = self.sam(**x, multimask_output=True)
        m = self.sam_proc.post_process_masks(out.pred_masks.cpu(), x["original_sizes"])[0][0]
        best = int(out.iou_scores[0, 0].argmax())
        return m[best].numpy() > 0.5

    def parse(self, img: Image.Image) -> dict[str, np.ndarray]:
        """Returns {region id: bool mask}; each pixel belongs to at most one region.

        Detection-instanced classes (cabinets, and furniture later) come out one region per box
        group, because the detector is the only thing that knows where one door ends and the next
        begins. Every other class comes out as a single `<kind>#0` that `Engine._instance` divides
        by geometry instead. `occluder` is one plain mask, never a region.
        """
        dets = self.detect(img)
        H, W = img.size[1], img.size[0]
        # drop boxes that cover ~whole image for small objects (DINO false positives)
        dets = [d for d in dets if not (d[0] == "occluder" and _box_area(d[2]) > 0.4 * H * W)]
        masks = self.masks_from_boxes(img, [d[2] for d in dets])
        # a cabinet box whose SAM mask fills <50 % of it is fragmented (stool legs, live-edge slabs,
        # shelving): painting it gives speckled patches, which look worse than leaving it untouched
        keep = [not (d[0] == "cabinet" and m.sum() < 0.5 * _box_area(d[2])) for d, m in zip(dets, masks)]
        dets = [d for d, k in zip(dets, keep) if k]
        masks = masks[np.array(keep, bool)] if len(keep) else masks
        # one label per detection (not per class): priority still decides who owns a contested
        # pixel, but which *box* won it is no longer thrown away
        label = np.full((H, W), -1, np.int32)
        best = np.zeros((H, W), np.float32)
        for i, (d, m) in enumerate(zip(dets, masks)):
            pr = _rank(d)
            win = m & (pr > best)
            label[win] = i
            best[win] = pr
        out = {}
        for kind, groups in group_detections(dets).items():
            if kind == "occluder":   # not a region: one mask, only ever used to block
                m = np.isin(label, [i for g in groups for i in g])
                out["occluder"] = m
                continue
            for gi, idxs in enumerate(groups):
                m = np.isin(label, idxs)
                if m.any():
                    out[make_rid(kind, gi)] = m
        return out


def _chunk(seq, size=MAX_PHRASES_PER_PASS):
    return [seq[i:i + size] for i in range(0, len(seq), size)] or [[]]


def prompt_passes() -> list[list[str]]:
    """The phrase list for each Grounding DINO pass, in order.

    The surface prompt that works on kitchens is kept intact; every group in `OWN_PASS` runs
    separately (see the note there), and the occluders — now long enough to cover rooms other than
    kitchens — are split into token-sized chunks. More phrases means more passes, so analysis time
    grows roughly linearly with this list.
    """
    own = {p for grp in OWN_PASS for p in grp if p in PROMPTS}
    surf = [p for p in PROMPTS if PROMPTS[p] != "occluder" and p not in own]
    occ = [p for p in PROMPTS if PROMPTS[p] == "occluder"]
    passes = _chunk(surf) + [list(g) for g in OWN_PASS] + _chunk(occ)
    return [p for p in passes if p]


def _is_run(det) -> bool:
    """A box matched by one of the whole-run phrases rather than a single door."""
    return any(w in det[3] for w in ("upper", "base"))


def _rank(det) -> float:
    """Label precedence for one detection: which box owns a contested pixel.

    Across classes this is `PRIORITY`, unchanged. Within a class an individual box always outranks
    a whole-run extender, whatever their confidences, so a run phrase can only claim the doors no
    single box found — which is the only reason it is in the prompt set. Both terms stay inside the
    class's own [PRIORITY, PRIORITY+1) band, so cross-class ordering is untouched.
    """
    return PRIORITY[det[0]] + (0.001 * det[1] if _is_run(det) else 0.1 + 0.9 * det[1])


def group_detections(dets) -> dict[str, list[list[int]]]:
    """Same-class detections that describe one physical thing, grouped — {class: [[idx, ...], ...]}.

    DINO fires several boxes on one cabinet door, so boxes that mostly cover each other are one
    region. The "upper/base cabinet" run phrases are extenders by design (`detect` keeps them only
    where they overlap a door box) and span several doors at once, so they are grouped separately
    from individual boxes — see the loop below. Classes that are not detection-instanced come back
    as a single group: their regions are decided by geometry, later.
    """
    groups = {}
    for kind in sorted({d[0] for d in dets}):
        idxs = [i for i, d in enumerate(dets) if d[0] == kind]
        if instancing(kind) != DETECTION:
            groups[kind] = [idxs]
            continue
        parent = {i: i for i in idxs}

        def find(a, parent=parent):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        for ai, a in enumerate(idxs):
            for b in idxs[ai + 1:]:
                # a run box never merges the doors it spans: doing so would collapse a whole
                # cabinet run into one region and defeat per-door addressing. It groups on its
                # own, and `_rank` then leaves it only the pixels no individual door claimed.
                if _is_run(dets[a]) != _is_run(dets[b]):
                    continue
                if _overlap(dets[a][2], dets[b][2]) > 0.5:
                    ra, rb = find(a), find(b)
                    if ra != rb:
                        parent[max(ra, rb)] = min(ra, rb)
        bins = {}
        for i in idxs:
            bins.setdefault(find(i), []).append(i)
        groups[kind] = [bins[k] for k in sorted(bins)]
    return groups


def _overlap(a, b):
    """Intersection area as a fraction of the smaller box."""
    iw = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    return iw * ih / max(1e-6, min(_box_area(a), _box_area(b)))


def _box_area(b):
    return max(0, b[2] - b[0]) * max(0, b[3] - b[1])


class Geometry:
    MODEL = "Ruicheng/moge-2-vitl-normal"

    def __init__(self, device):
        from moge.model.v2 import MoGeModel
        self.device = device
        self.model = MoGeModel.from_pretrained(self.MODEL).to(device).eval()

    @torch.no_grad()
    def solve(self, img: Image.Image) -> dict:
        x = torch.from_numpy(np.asarray(img, np.float32) / 255.0).permute(2, 0, 1).to(self.device)
        out = self.model.infer(x)
        H, W = x.shape[1:]
        K = out["intrinsics"].cpu().numpy().astype(np.float64)
        K[0] *= W  # MoGe returns intrinsics normalised by image size
        K[1] *= H
        return dict(points=out["points"].cpu().numpy(), valid=out["mask"].cpu().numpy() > 0.5,
                    K=K, normal=out["normal"].cpu().numpy() if "normal" in out else None)
