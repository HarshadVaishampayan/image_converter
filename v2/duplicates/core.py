"""Core duplicate-detection logic (no web code here).

Level 1  : file-name similarity + file-size match.
Level 2a : exact byte match (hash compare).
Level 2b : image similarity (embeddings via image_converter, see image_sim.py).
"""
import hashlib
import os
import re
from collections import defaultdict
from difflib import SequenceMatcher

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tif", ".tiff",
              ".webp", ".heic", ".heif"}

_EXT_ALIASES = {".jpeg": ".jpg", ".tif": ".tiff", ".heif": ".heic"}

# Suffixes OSes/apps add to copies: "foo (1)", "foo copy", "foo - Copy (2)", "foo_copy"
_COPY_MARKERS = re.compile(
    r"(\s*\(\d+\)|[\s_-]*copy(\s*\(?\d+\)?)?|[\s_-]+dup(licate)?\d*)+$",
    re.IGNORECASE,
)


from jobs import Cancelled  # noqa: E402  (shared with the job runner)


def is_image(path):
    return os.path.splitext(path)[1].lower() in IMAGE_EXTS


def normalize_name(filename):
    stem, ext = os.path.splitext(filename)
    ext = _EXT_ALIASES.get(ext.lower(), ext.lower())
    stem = _COPY_MARKERS.sub("", stem.strip()).strip().lower()
    stem = re.sub(r"[\s_.\-]+", " ", stem)
    return stem, ext


def name_similarity(a, b):
    """0..1 similarity of two file names, ignoring case and copy markers."""
    sa, ea = normalize_name(a)
    sb, eb = normalize_name(b)
    if sa == sb:
        score = 1.0
    else:
        m = SequenceMatcher(None, sa, sb, autojunk=False)
        score = m.ratio()
    if ea != eb:
        score *= 0.9  # same content can carry a different extension; mild penalty
    return score


def file_info(path, st=None):
    st = st or os.stat(path)
    return {
        "path": path,
        "name": os.path.basename(path),
        "size": st.st_size,
        "mtime": st.st_mtime,
        "is_image": is_image(path),
    }


def walk_files(root, include_hidden=False, skip_empty=True, min_size=0,
               progress=None, cancel=None):
    """Yield file_info dicts for every regular file under root (no symlinks)."""
    files = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        if cancel and cancel():
            raise Cancelled()
        if not include_hidden:
            dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        for fn in filenames:
            if not include_hidden and fn.startswith("."):
                continue
            p = os.path.join(dirpath, fn)
            try:
                st = os.lstat(p)
            except OSError:
                continue
            if not os.path.isfile(p) or os.path.islink(p):
                continue
            if skip_empty and st.st_size == 0:
                continue
            if st.st_size < min_size:
                continue
            files.append(file_info(p, st))
        if progress:
            progress(len(files), None, f"Indexed {len(files):,} files…")
    return files


def pair_key(a, b):
    return "\x00".join(sorted((a, b)))


def level1(files, name_threshold=0.7, size_tolerance_pct=0.0,
           progress=None, cancel=None):
    """Pairs whose sizes match (within tolerance) and whose names are similar."""
    files = sorted(files, key=lambda f: f["size"])
    tol = max(0.0, size_tolerance_pct) / 100.0
    pairs = []
    n = len(files)
    for i in range(n):
        if cancel and i % 200 == 0 and cancel():
            raise Cancelled()
        fi = files[i]
        limit = fi["size"] * (1 + tol)
        j = i + 1
        while j < n and files[j]["size"] <= limit:
            fj = files[j]
            score = name_similarity(fi["name"], fj["name"])
            if score >= name_threshold:
                pairs.append({
                    "a": fi, "b": fj,
                    "name_score": round(score, 4),
                    "size_match": fi["size"] == fj["size"],
                })
            j += 1
        if progress and i % 500 == 0:
            progress(i, n, f"Comparing names… {i:,}/{n:,}")
    pairs.sort(key=lambda p: (-p["name_score"], -p["a"]["size"]))
    return pairs


# ---------------------------------------------------------------- hashing ----

_hash_cache = {}  # (path, size, mtime) -> hexdigest
_CHUNK = 1 << 20


def _hash(path, limit=None):
    h = hashlib.sha256()
    read = 0
    with open(path, "rb") as fh:
        while True:
            want = _CHUNK if limit is None else min(_CHUNK, limit - read)
            if want <= 0:
                break
            buf = fh.read(want)
            if not buf:
                break
            h.update(buf)
            read += len(buf)
    return h.hexdigest()


def full_hash(path):
    st = os.stat(path)
    key = (path, st.st_size, st.st_mtime_ns)
    if key not in _hash_cache:
        _hash_cache[key] = _hash(path)
    return _hash_cache[key]


def bytes_equal(a, b):
    sa, sb = os.path.getsize(a), os.path.getsize(b)
    if sa != sb:
        return False
    return full_hash(a) == full_hash(b)


def verify_pairs_bytes(pairs, progress=None, cancel=None):
    """Annotate each pair with bytes_equal True/False (None if unreadable)."""
    for i, p in enumerate(pairs):
        if cancel and cancel():
            raise Cancelled()
        try:
            p["bytes_equal"] = bytes_equal(p["a"]["path"], p["b"]["path"])
        except OSError as e:
            p["bytes_equal"] = None
            p["error"] = str(e)
        if progress:
            progress(i + 1, len(pairs), f"Comparing bytes… {i + 1:,}/{len(pairs):,}")
    return pairs


def exact_dupes_tree(files, progress=None, cancel=None):
    """Whole-tree exact duplicates: size -> first 64KB hash -> full hash.

    Returns pairs (keeper = oldest file in each group, paired with every other member).
    """
    by_size = defaultdict(list)
    for f in files:
        by_size[f["size"]].append(f)
    candidates = [g for g in by_size.values() if len(g) > 1]
    total = sum(len(g) for g in candidates)
    done = 0
    pairs = []
    for group in candidates:
        by_partial = defaultdict(list)
        for f in group:
            if cancel and cancel():
                raise Cancelled()
            try:
                by_partial[_hash(f["path"], limit=65536)].append(f)
            except OSError:
                pass
            done += 1
            if progress:
                progress(done, total, f"Hashing… {done:,}/{total:,} candidate files")
        for sub in by_partial.values():
            if len(sub) < 2:
                continue
            by_full = defaultdict(list)
            for f in sub:
                try:
                    by_full[full_hash(f["path"])].append(f)
                except OSError:
                    pass
            for members in by_full.values():
                if len(members) < 2:
                    continue
                members.sort(key=lambda f: f["mtime"])
                keeper = members[0]
                for other in members[1:]:
                    pairs.append({
                        "a": keeper, "b": other,
                        "name_score": round(name_similarity(keeper["name"], other["name"]), 4),
                        "size_match": True,
                        "bytes_equal": True,
                    })
    return pairs
