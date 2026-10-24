"""Image similarity using the image_converter project's embedding models.

Uses v2/model_manager.py to load the same SigLIP 2 / CLIP models the search
index uses, then compares images by cosine similarity of their embeddings.
"""
import os
import sys
import threading

import numpy as np

from .core import Cancelled, name_similarity

_V2_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_lock = threading.Lock()
_manager = None
_emb_cache = {}  # (model, path, size, mtime_ns) -> np.ndarray (unit length)
_import_error = None

try:
    from PIL import Image, ImageOps
    try:
        from pillow_heif import register_heif_opener
        register_heif_opener()
    except ImportError:
        pass
except ImportError as e:  # pragma: no cover
    _import_error = str(e)


def available():
    """(ok, message) — whether the ML stack can be imported."""
    try:
        import torch  # noqa: F401
        import sentence_transformers  # noqa: F401
        import transformers  # noqa: F401
    except Exception as e:
        return False, f"Image similarity needs torch/transformers/sentence-transformers: {e}"
    return True, "ok"


def _get_model(model_type):
    global _manager
    with _lock:
        if _manager is None:
            if _V2_DIR not in sys.path:
                sys.path.insert(0, _V2_DIR)
            from model_manager import ModelManager
            _manager = ModelManager()
        return _manager.get_embedding_model(model_type)


def _load(path, max_side=512):
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    img.draft("RGB", (max_side, max_side))
    img = img.convert("RGB")
    img.thumbnail((max_side, max_side))
    return img


def embed_paths(paths, model_type="siglip", batch_size=16, progress=None, cancel=None):
    """Return {path: unit embedding} for every image that could be read."""
    out = {}
    todo = []
    for p in paths:
        try:
            st = os.stat(p)
        except OSError:
            continue
        key = (model_type, p, st.st_size, st.st_mtime_ns)
        if key in _emb_cache:
            out[p] = _emb_cache[key]
        else:
            todo.append((p, key))

    if todo:
        if progress:
            progress(0, len(todo), f"Loading {model_type.upper()} model (first run downloads it)…")
        model = _get_model(model_type)

    done = 0
    for i in range(0, len(todo), batch_size):
        if cancel and cancel():
            raise Cancelled()
        chunk = todo[i:i + batch_size]
        imgs, keys = [], []
        for p, key in chunk:
            try:
                imgs.append(_load(p))
                keys.append((p, key))
            except Exception:
                pass  # unreadable / unsupported image: skip
        if imgs:
            with _lock:  # models aren't thread-safe
                feats = model.encode(imgs, convert_to_tensor=True)
            arr = feats.float().cpu().numpy()
            if arr.ndim == 1:
                arr = arr[None, :]
            arr /= np.linalg.norm(arr, axis=1, keepdims=True).clip(min=1e-12)
            for (p, key), vec in zip(keys, arr):
                _emb_cache[key] = vec
                out[p] = vec
        done += len(chunk)
        if progress:
            progress(done, len(todo), f"Embedding images… {done:,}/{len(todo):,}")
    return out


def compare_pairs(pairs, model_type="siglip", progress=None, cancel=None):
    """Annotate pairs where both sides are images with image_sim (0..1)."""
    img_pairs = [p for p in pairs if p["a"]["is_image"] and p["b"]["is_image"]]
    paths = sorted({p["a"]["path"] for p in img_pairs} | {p["b"]["path"] for p in img_pairs})
    embs = embed_paths(paths, model_type, progress=progress, cancel=cancel)
    for p in img_pairs:
        ea, eb = embs.get(p["a"]["path"]), embs.get(p["b"]["path"])
        if ea is not None and eb is not None:
            p["image_sim"] = round(float(np.dot(ea, eb)), 4)
            p["image_model"] = model_type
    return img_pairs


def similar_images_tree(files, threshold=0.92, model_type="siglip", max_pairs=5000,
                        progress=None, cancel=None):
    """Whole-tree near-duplicate images: all-vs-all cosine similarity, chunked."""
    imgs = [f for f in files if f["is_image"]]
    embs = embed_paths([f["path"] for f in imgs], model_type, progress=progress, cancel=cancel)
    imgs = [f for f in imgs if f["path"] in embs]
    if len(imgs) < 2:
        return []
    mat = np.stack([embs[f["path"]] for f in imgs]).astype(np.float32)
    pairs = []
    block = 1024
    n = len(imgs)
    for start in range(0, n, block):
        if cancel and cancel():
            raise Cancelled()
        sims = mat[start:start + block] @ mat.T
        rows, cols = np.nonzero(sims >= threshold)
        for r, c in zip(rows, cols):
            i = start + r
            if c <= i:
                continue  # upper triangle only, skip self
            pairs.append((float(sims[r, c]), i, int(c)))
        if progress:
            progress(min(start + block, n), n, "Comparing image embeddings…")
    pairs.sort(reverse=True)
    result = []
    for sim, i, j in pairs[:max_pairs]:
        a, b = imgs[i], imgs[j]
        result.append({
            "a": a, "b": b,
            "name_score": round(name_similarity(a["name"], b["name"]), 4),
            "size_match": a["size"] == b["size"],
            "image_sim": round(sim, 4),
            "image_model": model_type,
        })
    return result
