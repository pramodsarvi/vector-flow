"""Bulk catalogue import / export as a single ZIP.

Format (what `export_zip` writes, and what `plan_import` accepts):

    catalog.json                 list of {id, name, category, vendor, price, tile_w, tile_h, gloss}
    <id>/albedo.png              seamless colour texture (required for new SKUs)
    <id>/normal.png              optional, OpenGL tangent-space normal map
    <id>/roughness.png           optional, grayscale roughness

Instead of catalog.json a `catalog.csv` with the same eight columns may be used (comma, semicolon or tab
separated, UTF-8 with or without BOM). Optional `albedo` / `normal` / `roughness` columns name image files
inside the ZIP; without them `<id>/<map>.(png|jpg|jpeg|webp)` is looked up (case-insensitive). A wrapper
folder, `__MACOSX/` and dot-files are ignored.

Import is all-or-nothing: every row and image is validated first and *all* problems are reported with row
numbers; if there is any error nothing changes. Modes:
  merge   (default) add new SKUs, update listed ones; an existing SKU without a texture in the upload keeps
          its texture (so a CSV alone can update prices); unlisted SKUs stay.
  replace the catalogue becomes exactly the upload; unlisted SKUs are removed.
Replacing an albedo without supplying new maps drops the old normal/roughness maps (they'd no longer match).

Applying is crash-safe: new SKU folders are staged inside the catalogue directory (same filesystem, also
when it is a Docker volume mount point), swapped in, then catalog.json is written atomically, then removed
SKUs are deleted. `recover(root)` (called at server start) finishes or rolls back an interrupted apply.
"""
from __future__ import annotations

import csv
import io
import json
import re
import shutil
import uuid
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

FIELDS = ("id", "name", "category", "vendor", "price", "tile_w", "tile_h", "gloss")
MAPS = {"albedo": "RGB", "normal": "RGB", "roughness": "L"}
IMAGE_EXT = (".png", ".jpg", ".jpeg", ".webp")
ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9-]{1,31}$")

MAX_ENTRIES = 20_000                 # zip members
MAX_TOTAL_BYTES = 8 * 2 ** 30        # declared uncompressed total
MAX_MEMBER_BYTES = 128 * 2 ** 20     # one image (bounded read: declared sizes can lie)
MAX_PIXELS = 64_000_000
MAX_SIDE = 2048                      # stored textures are downscaled to this
MIN_SIDE = 64


class CatalogImportError(Exception):
    def __init__(self, errors: list[str]):
        super().__init__("; ".join(errors[:5]) + (" …" if len(errors) > 5 else ""))
        self.errors = errors


def validate_entry(d: dict, categories) -> list[str]:
    """Field checks shared by the single-SKU API and bulk import. Returns problems (empty = ok)."""
    errs = []
    if not ID_RE.match(str(d.get("id", ""))):
        errs.append("id must be 2–32 letters, digits or '-' (starting with a letter or digit)")
    if d.get("category") not in categories:
        errs.append(f"category must be one of {', '.join(categories)}")
    name = str(d.get("name", "")).strip()
    if not name or len(name) > 80:
        errs.append("name is required (≤ 80 characters)")
    if len(str(d.get("vendor", ""))) > 60:
        errs.append("vendor ≤ 60 characters")
    try:
        if not 0 <= float(d["price"]) <= 1e6:
            errs.append("price must be 0–1,000,000")
        if not (0.01 <= float(d["tile_w"]) <= 20 and 0.01 <= float(d["tile_h"]) <= 20):
            errs.append("tile_w / tile_h must be 0.01–20 m")
        if not 0 <= float(d["gloss"]) <= 1:
            errs.append("gloss must be 0–1")
    except (KeyError, TypeError, ValueError):
        errs.append("price, tile_w, tile_h and gloss must be numbers")
    return errs


@dataclass
class Plan:
    mode: str
    entries: list[dict] = field(default_factory=list)          # the resulting catalog.json
    files: dict[str, dict[str, str]] = field(default_factory=dict)   # id → {map: zip member}
    added: list[str] = field(default_factory=list)
    updated: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> dict:
        return dict(mode=self.mode, ok=not self.errors, added=self.added, updated=self.updated,
                    removed=self.removed, total=len(self.entries), errors=self.errors[:200],
                    error_count=len(self.errors), warnings=self.warnings[:50])


# ------------------------------------------------------------------ reading the archive
def _norm(name: str) -> str:
    n = name.replace("\\", "/")
    while n.startswith(("./", "/")):
        n = n[2:] if n.startswith("./") else n[1:]
    return n


def _junk(name: str) -> bool:
    parts = name.split("/")
    return parts[0] == "__MACOSX" or any(p.startswith(".") for p in parts) or parts[-1].lower() == "thumbs.db"


def _read_member(zf: zipfile.ZipFile, info: zipfile.ZipInfo) -> bytes:
    with zf.open(info) as f:
        data = f.read(MAX_MEMBER_BYTES + 1)
    if len(data) > MAX_MEMBER_BYTES:
        raise ValueError(f"{info.filename}: larger than {MAX_MEMBER_BYTES // 2 ** 20} MB")
    return data


def _num(v, default=None):
    if v is None or (isinstance(v, str) and not v.strip()):
        if default is None:
            raise ValueError
        return default
    return float(str(v).strip().replace(",", ".")) if isinstance(v, str) else float(v)


def _parse_manifest(raw: bytes, is_csv: bool) -> tuple[list[dict], list[str]]:
    """Rows as dicts with lower-cased, stripped keys + parse errors."""
    text = raw.decode("utf-8-sig")
    if not is_csv:
        data = json.loads(text)
        if isinstance(data, dict):
            data = data.get("skus") or data.get("catalog") or []
        if not isinstance(data, list):
            return [], ["catalog.json must be a list of SKU objects"]
        return [{str(k).strip().lower(): v for k, v in r.items()} if isinstance(r, dict) else {"__bad__": r}
                for r in data], []
    sample = text[:4096]
    try:
        delim = csv.Sniffer().sniff(sample, delimiters=",;\t").delimiter
    except csv.Error:
        delim = ";" if sample.count(";") > sample.count(",") else ","
    rows = list(csv.DictReader(io.StringIO(text), delimiter=delim))
    return [{(k or "").strip().lower(): (v.strip() if isinstance(v, str) else v) for k, v in r.items()}
            for r in rows if any((v or "").strip() for v in r.values() if isinstance(v, str))], []


def plan_import(zip_path: Path, root: Path, categories, mode: str = "merge") -> Plan:
    """Validate an archive against the current catalogue at `root`; decodes every referenced image.
    Never writes anything."""
    plan = Plan(mode=mode)
    if mode not in ("merge", "replace"):
        plan.errors.append("mode must be 'merge' or 'replace'")
        return plan
    try:
        zf = zipfile.ZipFile(zip_path)
    except (zipfile.BadZipFile, OSError):
        plan.errors.append("not a valid ZIP file")
        return plan
    with zf:
        infos = [i for i in zf.infolist() if not i.is_dir() and not _junk(_norm(i.filename))]
        if len(infos) > MAX_ENTRIES:
            plan.errors.append(f"too many files in the archive (> {MAX_ENTRIES})")
            return plan
        if sum(i.file_size for i in infos) > MAX_TOTAL_BYTES:
            plan.errors.append(f"archive expands to more than {MAX_TOTAL_BYTES // 2 ** 30} GB")
            return plan
        # manifest: shallowest catalog.json (preferred) or catalog.csv; its folder is the archive root
        manifests = [i for i in infos if _norm(i.filename).split("/")[-1].lower() in ("catalog.json", "catalog.csv")]
        if not manifests:
            plan.errors.append("archive must contain catalog.json or catalog.csv")
            return plan
        manifests.sort(key=lambda i: (_norm(i.filename).count("/"), not i.filename.lower().endswith(".json")))
        man = manifests[0]
        prefix = _norm(man.filename).rsplit("/", 1)[0] + "/" if "/" in _norm(man.filename) else ""
        index = {}
        for i in infos:
            n = _norm(i.filename)
            if n.lower().startswith(prefix.lower()):
                index.setdefault(n[len(prefix):].lower(), i)
        try:
            rows, perr = _parse_manifest(_read_member(zf, man), man.filename.lower().endswith(".csv"))
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError, csv.Error) as e:
            plan.errors.append(f"{man.filename}: cannot read ({e})")
            return plan
        plan.errors += perr
        if not rows and not perr:
            plan.errors.append(f"{man.filename}: no SKUs listed")

        current = json.loads((root / "catalog.json").read_text()) if (root / "catalog.json").exists() else []
        cur_by_id = {e["id"]: e for e in current}
        cur_lower = {e["id"].lower(): e["id"] for e in current}
        seen: dict[str, int] = {}
        line0 = 2 if man.filename.lower().endswith(".csv") else 1     # CSV row 1 is the header
        for n, r in enumerate(rows):
            where = f"row {n + line0}"
            if "__bad__" in r:
                plan.errors.append(f"{where}: not an object")
                continue
            sid = str(r.get("id", "")).strip()
            where += f" ({sid})" if sid else ""
            try:
                d = dict(id=sid, name=str(r.get("name", "")).strip(),
                         category=str(r.get("category", "")).strip().lower(),
                         vendor=str(r.get("vendor", "") or "").strip(), price=_num(r.get("price")),
                         tile_w=_num(r.get("tile_w")), tile_h=_num(r.get("tile_h")), gloss=_num(r.get("gloss"), 0.3))
            except ValueError:
                plan.errors.append(f"{where}: price, tile_w and tile_h are required numbers (gloss optional)")
                continue
            plan.errors += [f"{where}: {e}" for e in validate_entry(d, categories)]
            low = sid.lower()
            if low in seen:
                plan.errors.append(f"{where}: duplicate id (also row {seen[low]})")
                continue
            seen[low] = n + line0
            if low in cur_lower and cur_lower[low] != sid:
                plan.errors.append(f"{where}: id differs only by case from existing '{cur_lower[low]}'")
                continue
            # image references
            refs = {}
            for kind in MAPS:
                explicit = str(r.get(kind, "") or "").strip()
                if explicit:
                    m = index.get(_norm(explicit).lower())
                    if m is None:
                        plan.errors.append(f"{where}: {kind} file '{explicit}' not found in the archive")
                        continue
                    refs[kind] = m.filename
                else:
                    for ext in IMAGE_EXT:
                        m = index.get(f"{low}/{kind}{ext}")
                        if m is not None:
                            refs[kind] = m.filename
                            break
            exists = sid in cur_by_id
            if "albedo" not in refs and not exists:
                plan.errors.append(f"{where}: new SKU needs {sid}/albedo.png (or an 'albedo' column)")
            if "albedo" not in refs and exists and not (root / sid / "albedo.png").exists():
                plan.errors.append(f"{where}: existing SKU has no texture on the server; include albedo")
            for kind, member in refs.items():
                try:
                    _decode(_read_member(zf, zf.getinfo(member)), kind)
                except ValueError as e:
                    plan.errors.append(f"{where}: {kind}: {e}")
            if refs:
                plan.files[sid] = refs
            extra = set(r) - set(FIELDS) - set(MAPS)
            if extra and n == 0:
                plan.warnings.append(f"ignored columns: {', '.join(sorted(extra))}")
            plan.entries.append(d)
            (plan.updated if exists else plan.added).append(sid)

        listed = {e["id"] for e in plan.entries}
        if mode == "merge":
            plan.entries = [cur_by_id[c] if c not in listed else next(e for e in plan.entries if e["id"] == c)
                            for c in cur_by_id] + [e for e in plan.entries if e["id"] not in cur_by_id]
        else:
            plan.removed = [c for c in cur_by_id if c not in listed]
        unlisted_dirs = {_norm(i.filename)[len(prefix):].split("/")[0] for i in infos
                         if "/" in _norm(i.filename)[len(prefix):]}
        orphans = sorted(x for x in unlisted_dirs if x.lower() not in seen)
        if orphans:
            plan.warnings.append(f"folders not listed in the manifest were ignored: {', '.join(orphans[:10])}")
    return plan


def _decode(raw: bytes, kind: str) -> Image.Image:
    try:
        im = Image.open(io.BytesIO(raw))
        if im.width * im.height > MAX_PIXELS:
            raise ValueError(f"image too large ({im.width}×{im.height})")
        im.load()
    except ValueError:
        raise
    except Exception:  # noqa: BLE001  (truncated / not an image / decompression bomb)
        raise ValueError("not a readable image") from None
    if min(im.size) < MIN_SIDE:
        raise ValueError(f"image must be at least {MIN_SIDE} px")
    return im.convert(MAPS[kind])


# ------------------------------------------------------------------ applying
def apply_import(zip_path: Path, root: Path, plan: Plan, _fail_after_swaps: int | None = None) -> None:
    """Apply a validated plan. Re-reads images from the archive (plan holds no pixel data)."""
    if plan.errors:
        raise CatalogImportError(plan.errors)
    root.mkdir(parents=True, exist_ok=True)
    stage = root / f".import-{uuid.uuid4().hex[:8]}"
    stage.mkdir()
    with zipfile.ZipFile(zip_path) as zf:
        for sid, refs in plan.files.items():
            d = stage / sid
            d.mkdir()
            if "albedo" not in refs and (root / sid / "albedo.png").exists():
                shutil.copy2(root / sid / "albedo.png", d / "albedo.png")      # maps-only update: keep texture
                for k in ("normal", "roughness"):
                    if k not in refs and (root / sid / f"{k}.png").exists():
                        shutil.copy2(root / sid / f"{k}.png", d / f"{k}.png")
            for kind, member in refs.items():
                im = _decode(_read_member(zf, zf.getinfo(member)), kind)
                im.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
                im.save(d / f"{kind}.png")
    (stage / "catalog.json").write_text(json.dumps(plan.entries, indent=2))
    (stage / "removed.json").write_text(json.dumps(plan.removed))
    (stage / "READY").write_text("")      # from here on recover() rolls *forward*
    _commit(root, stage, _fail_after_swaps)


def _commit(root: Path, stage: Path, _fail_after_swaps=None):
    swapped = 0
    for d in sorted(p for p in stage.iterdir() if p.is_dir() and not p.name.startswith(".")):
        sid = d.name
        if (root / sid).exists():
            (root / sid).rename(stage / f".old-{sid}")
        d.rename(root / sid)
        swapped += 1
        if _fail_after_swaps is not None and swapped >= _fail_after_swaps:
            raise RuntimeError("simulated crash during apply")
    tmp = root / "catalog.json.tmp"
    shutil.copy2(stage / "catalog.json", tmp)
    tmp.replace(root / "catalog.json")
    for sid in json.loads((stage / "removed.json").read_text()):
        if ID_RE.match(sid) and (root / sid).is_dir():
            shutil.rmtree(root / sid)
    shutil.rmtree(stage)


def recover(root: Path) -> list[str]:
    """Finish (READY) or roll back (not READY) any interrupted import. Safe to call on every start."""
    done = []
    for stage in root.glob(".import-*"):
        if (stage / "READY").exists():
            _commit(root, stage)                       # idempotent: skips what was already swapped
            done.append(f"{stage.name}: completed")
        else:
            shutil.rmtree(stage)
            done.append(f"{stage.name}: discarded (validation/staging never finished)")
    return done


# ------------------------------------------------------------------ export
def export_zip(root: Path, out: Path) -> Path:
    """Write the current catalogue as a ZIP that `plan_import` accepts unchanged."""
    entries = json.loads((root / "catalog.json").read_text())
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_STORED) as zf:   # PNGs are already compressed
        zf.writestr("catalog.json", json.dumps([{k: e[k] for k in FIELDS} for e in entries], indent=2))
        for e in entries:
            for kind in MAPS:
                p = root / e["id"] / f"{kind}.png"
                if p.exists():
                    zf.write(p, f"{e['id']}/{kind}.png")
    return out
