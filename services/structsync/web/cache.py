"""Disposable on-disk result cache for the web app.

Two levels, both under STRUCTSYNC_DATA/cache (delete the folder any time — projects never depend on it):

  analysis/<key>.npz + .json   key = photo bytes + code/model version
      Same photo uploaded again → detection + geometry (the 10–30 s step) is a file copy.
  renders/<key>.jpg + .json    key = photo bytes + scene revision (analysis + tap corrections) + materials + options
                               + each SKU's catalogue entry and texture *pixels* + code version
      Same photo + same style → the stored image and report are returned without rendering.

Keys never need a model or a loaded Scene. The code version hashes every structsync/*.py source plus the
model ids, so any code or model change invalidates automatically. Texture identity hashes decoded pixels,
not file bytes or mtimes, so re-importing an unchanged catalogue export keeps the cache warm.
Note: a cached render's SKU-recognition report reflects the catalogue at the time it was first rendered.

Size cap STRUCTSYNC_CACHE_MB (default 4096; 0 disables): oldest entries (by last use) are pruned.
All calls happen on the single engine worker, so no locking is needed.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

PKG = Path(__file__).resolve().parent.parent / "structsync"


def _sha(*parts: bytes) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p)
        h.update(b"\0")
    return h.hexdigest()


def code_version() -> str:
    """Hash of the pipeline sources + model ids (incl. env overrides)."""
    from structsync.catalog import CatalogSearch
    from structsync.vision import Geometry, Segmenter
    srcs = [p.read_bytes() for p in sorted(PKG.glob("*.py"))]
    models = f"{Segmenter.DINO}|{Segmenter.SAM}|{Geometry.MODEL}|{CatalogSearch.MODEL}".encode()
    return _sha(*srcs, models)[:16]


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(2 ** 20):
            h.update(chunk)
    return h.hexdigest()


def scene_rev(state: dict) -> str:
    """Revision of an analysed scene (geometry + every surface mask): changes with each tap correction."""
    parts = []
    for k in sorted(state):
        a = np.ascontiguousarray(np.asarray(state[k]))
        if a.dtype == bool:
            a = np.packbits(a)
        parts += [k.encode(), str(a.dtype).encode(), str(a.shape).encode(), a.tobytes()]
    return _sha(*parts)[:24]


class Cache:
    def __init__(self, root: Path, limit_mb: int | None = None):
        self.root = Path(root)
        self.limit = (int(os.environ.get("STRUCTSYNC_CACHE_MB", "4096")) if limit_mb is None else limit_mb) * 2 ** 20
        self.version = code_version()
        self._tex: dict[tuple, str] = {}     # (path, mtime_ns, size) → pixel hash

    @property
    def enabled(self) -> bool:
        return self.limit > 0

    def _dir(self, kind: str) -> Path:
        d = self.root / kind
        d.mkdir(parents=True, exist_ok=True)
        return d

    # ------------------------------------------------------------ keys
    def analysis_key(self, photo_sha: str) -> str:
        return _sha(b"analysis", photo_sha.encode(), self.version.encode())[:32]

    def texture_sha(self, path: Path) -> str:
        """Hash of decoded pixels (stable across PNG re-encodes), memoised per file version."""
        st = path.stat()
        k = (str(path), st.st_mtime_ns, st.st_size)
        if k not in self._tex:
            im = Image.open(path)
            self._tex[k] = _sha(im.mode.encode(), str(im.size).encode(), im.tobytes())
        return self._tex[k]

    def render_key(self, rev: str, assign: dict, options: dict, skus: dict) -> str:
        """skus: {id: SKU}. Includes the full catalogue entry (the cached record carries the BOM)."""
        sku_part = {}
        for sid in sorted(set(assign.values())):
            s = skus[sid]
            # numbers normalised: an import writes 70.0 where the old JSON said 70 — same SKU
            entry = dict(id=s.id, name=s.name, category=s.category, vendor=s.vendor, price=float(s.price),
                         tile_w=float(s.tile_w), tile_h=float(s.tile_h), gloss=float(s.gloss))
            entry["maps"] = {k: self.texture_sha(s.path / f"{k}.png") for k in ("albedo", "normal", "roughness")
                             if (s.path / f"{k}.png").exists()}
            sku_part[sid] = entry
        opts = dict(options)
        if not opts.get("harmonize"):
            opts.pop("strength", None)
        blob = json.dumps(dict(rev=rev, assign=dict(sorted(assign.items())), options=dict(sorted(opts.items())),
                               skus=sku_part, version=self.version), sort_keys=True, default=str)
        return _sha(b"render", blob.encode())[:32]

    # ------------------------------------------------------------ entries
    def _touch(self, *paths: Path):
        for p in paths:
            try:
                os.utime(p)
            except OSError:
                pass

    def get_analysis(self, key: str):
        if not self.enabled:
            return None
        d = self._dir("analysis")
        npz, meta = d / f"{key}.npz", d / f"{key}.json"
        if not (npz.exists() and meta.exists()):
            return None
        self._touch(npz, meta)
        return npz, json.loads(meta.read_text())

    def put_analysis(self, key: str, npz_src: Path, meta: dict):
        if not self.enabled:
            return
        d = self._dir("analysis")
        tmp = d / f".{key}.npz.tmp"
        shutil.copyfile(npz_src, tmp)
        tmp.replace(d / f"{key}.npz")                 # data first, JSON last: JSON marks the entry complete
        self._write_json(d / f"{key}.json", meta)
        self.prune()

    def get_render(self, key: str):
        if not self.enabled:
            return None
        d = self._dir("renders")
        jpg, meta = d / f"{key}.jpg", d / f"{key}.json"
        if not (jpg.exists() and meta.exists()):
            return None
        self._touch(jpg, meta)
        return jpg, json.loads(meta.read_text())

    def put_render(self, key: str, jpg_src: Path, record: dict):
        if not self.enabled:
            return
        d = self._dir("renders")
        tmp = d / f".{key}.jpg.tmp"
        shutil.copyfile(jpg_src, tmp)
        tmp.replace(d / f"{key}.jpg")
        self._write_json(d / f"{key}.json", record)
        self.prune()

    @staticmethod
    def _write_json(p: Path, data: dict):
        tmp = p.with_name(f".{p.name}.tmp")
        tmp.write_text(json.dumps(data))
        tmp.replace(p)

    # ------------------------------------------------------------ housekeeping
    def stats(self) -> dict:
        out = dict(enabled=self.enabled, limit_mb=self.limit // 2 ** 20, version=self.version)
        for kind in ("analysis", "renders"):
            files = [p for p in self._dir(kind).glob("*.json")]
            out[kind] = len(files)
        out["size_mb"] = round(sum(p.stat().st_size for p in self.root.rglob("*") if p.is_file()) / 2 ** 20, 1) \
            if self.root.exists() else 0
        return out

    def clear(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def prune(self):
        """Drop least-recently-used entries until the cache fits the size cap."""
        entries = []
        for kind, ext in (("analysis", ".npz"), ("renders", ".jpg")):
            for meta in self._dir(kind).glob("*.json"):
                data = meta.with_suffix(ext)
                files = [f for f in (meta, data) if f.exists()]
                entries.append((max(f.stat().st_mtime for f in files), sum(f.stat().st_size for f in files), files))
        for p in self.root.rglob(".*.tmp"):          # leftovers of an interrupted write
            p.unlink(missing_ok=True)
        total = sum(e[1] for e in entries)
        for _, size, files in sorted(entries, key=lambda e: e[0]):
            if total <= self.limit:
                break
            for f in files:
                f.unlink(missing_ok=True)
            total -= size
