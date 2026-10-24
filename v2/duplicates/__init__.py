"""Duplicates — Flask blueprint for finding duplicate files under a folder.

Level 1  : file-name similarity + file-size match           (core.level1)
Level 2a : exact byte match                                 (core.verify_pairs_bytes / exact_dupes_tree)
Level 2b : image similarity via ModelManager embeddings     (image_sim)

Mounted at /duplicates by gui.py.
"""
import io
import os
import subprocess
import threading

from flask import Blueprint, abort, current_app, jsonify, render_template, request, send_file

import image_ops
from find_similar import invalidate_cache

from jobs import JobRunner

from . import core, image_sim

bp = Blueprint("duplicates", __name__, url_prefix="/duplicates")

# ------------------------------------------------------------------ state ----
state_lock = threading.Lock()
state = {"root": None, "files": None, "pairs": {}}  # pairs: key -> pair dict
runner = JobRunner()  # one scan at a time


def _merge_pairs(new_pairs, source):
    with state_lock:
        for p in new_pairs:
            key = core.pair_key(p["a"]["path"], p["b"]["path"])
            existing = state["pairs"].get(key)
            if existing is None:
                p["key"] = key
                p["found_by"] = [source]
                state["pairs"][key] = p
            else:
                for field in ("bytes_equal", "image_sim", "image_model", "error"):
                    if field in p:
                        existing[field] = p[field]
                if source not in existing["found_by"]:
                    existing["found_by"].append(source)


def _start_job(kind, fn):
    job = runner.start(kind, fn)
    if job is None:
        abort(409, "Another scan is already running.")
    return jsonify(job.to_json())


def _ensure_index(job, root, opts=None):
    with state_lock:
        if state["root"] == root and state["files"] is not None:
            return state["files"]
    opts = opts or {}
    files = core.walk_files(root, progress=job.progress,
                            cancel=lambda: job.cancelled, **opts)
    with state_lock:
        state["root"], state["files"] = root, files
    return files


# --------------------------------------------------------------- security ----
@bp.before_request
def _guard():
    # Block cross-site form posts: require a custom header (forces a CORS
    # preflight, which this server never approves).
    if request.method == "POST" and request.headers.get("X-Requested-By") != "image-converter":
        abort(403)


def _within_root(path):
    root = state["root"]
    if not root or not path:
        return False
    real = os.path.realpath(path)
    # Reject anything that resolves elsewhere (symlinks, "..") or sits outside root.
    return real == os.path.abspath(path) and \
        os.path.commonpath([real, os.path.realpath(root)]) == os.path.realpath(root)


def _known_path(path):
    with state_lock:
        return any(path in (p["a"]["path"], p["b"]["path"]) for p in state["pairs"].values())


# ----------------------------------------------------------------- routes ----
@bp.get("/")
def page():
    ok, msg = image_sim.available()
    return render_template("duplicates.html", active="duplicates", image_ok=ok, image_msg=msg)


@bp.get("/api/status")
def status():
    ok, msg = image_sim.available()
    return jsonify({"image_ok": ok, "image_msg": msg, "root": state["root"],
                    "pair_count": len(state["pairs"])})


@bp.post("/api/scan/level1")
def scan_level1():
    body = request.get_json(force=True)
    root = os.path.realpath(os.path.expanduser(body.get("root", "").strip()))
    if not os.path.isdir(root):
        abort(400, "Folder does not exist.")
    opts = {"include_hidden": bool(body.get("include_hidden", False)),
            "skip_empty": bool(body.get("skip_empty", True)),
            "min_size": int(body.get("min_size", 0) or 0)}
    name_t = float(body.get("name_threshold", 0.7))
    tol = float(body.get("size_tolerance_pct", 0))

    def work(job):
        with state_lock:
            state["root"], state["files"], state["pairs"] = root, None, {}
        files = _ensure_index(job, root, opts)
        pairs = core.level1(files, name_t, tol, progress=job.progress,
                            cancel=lambda: job.cancelled)
        _merge_pairs(pairs, "level1")
        return len(pairs)

    return _start_job("level1", work)


@bp.post("/api/scan/bytes")
def scan_bytes():
    body = request.get_json(force=True)
    scope = body.get("scope", "pairs")
    if not state["root"]:
        abort(400, "Run Level 1 first (it sets the folder).")

    def work(job):
        cancel = lambda: job.cancelled  # noqa: E731
        if scope == "tree":
            files = _ensure_index(job, state["root"])
            pairs = core.exact_dupes_tree(files, progress=job.progress, cancel=cancel)
            _merge_pairs(pairs, "bytes")
            return len(pairs)
        with state_lock:
            pairs = list(state["pairs"].values())
        core.verify_pairs_bytes(pairs, progress=job.progress, cancel=cancel)
        return sum(1 for p in pairs if p.get("bytes_equal"))

    return _start_job("bytes", work)


@bp.post("/api/scan/images")
def scan_images():
    ok, msg = image_sim.available()
    if not ok:
        abort(400, msg)
    body = request.get_json(force=True)
    scope = body.get("scope", "pairs")
    model = body.get("model", "siglip")
    if model not in ("clip", "siglip"):
        abort(400, "model must be clip or siglip")
    threshold = float(body.get("threshold", 0.92))
    if not state["root"]:
        abort(400, "Run Level 1 first (it sets the folder).")

    def work(job):
        cancel = lambda: job.cancelled  # noqa: E731
        if scope == "tree":
            files = _ensure_index(job, state["root"])
            pairs = image_sim.similar_images_tree(files, threshold, model,
                                                  progress=job.progress, cancel=cancel)
            _merge_pairs(pairs, "image")
            return len(pairs)
        with state_lock:
            pairs = list(state["pairs"].values())
        compared = image_sim.compare_pairs(pairs, model, progress=job.progress, cancel=cancel)
        return sum(1 for p in compared if p.get("image_sim", 0) >= threshold)

    return _start_job("images", work)


@bp.get("/api/job/<job_id>")
def job_status(job_id):
    job = runner.get(job_id) or abort(404)
    return jsonify(job.to_json())


@bp.post("/api/job/<job_id>/cancel")
def job_cancel(job_id):
    job = runner.get(job_id) or abort(404)
    job.cancelled = True
    return jsonify(job.to_json())


def _index():
    return current_app.extensions.get("image_index")


@bp.get("/api/pairs")
def get_pairs():
    with state_lock:
        pairs = list(state["pairs"].values())
    api = _index()
    indexed = image_ops.indexed_paths(api) if api is not None and pairs else set()
    for p in pairs:  # whether "Forget from index" applies to each side
        p["a"]["indexed"] = p["a"]["path"] in indexed
        p["b"]["indexed"] = p["b"]["path"] in indexed
    return jsonify({"root": state["root"], "pairs": pairs})


def _result_file(body):
    path = (body or {}).get("path", "")
    if not (_within_root(path) and _known_path(path) and image_ops.safe_file(path)):
        abort(400, "Path is not a file from the current scan results.")
    return path


def _drop_from_results(path):
    with state_lock:
        state["pairs"] = {k: p for k, p in state["pairs"].items()
                          if path not in (p["a"]["path"], p["b"]["path"])}
        if state["files"] is not None:
            state["files"] = [f for f in state["files"] if f["path"] != path]


@bp.post("/api/delete")
def delete_file():
    """Trash a file from the results (409 + needs_permanent if there's no Trash),
    and forget it from the search index if it was indexed."""
    body = request.get_json(force=True)
    path = _result_file(body)
    try:
        how = image_ops.delete_file(path, permanent=bool(body.get("permanent")))
    except image_ops.TrashUnavailable as e:
        return jsonify({"error": f"This file can't be moved to the Trash ({e}).",
                        "needs_permanent": True, "path": path}), 409
    except OSError as e:
        abort(500, f"Couldn't delete the file: {e}")
    _drop_from_results(path)
    api = _index()
    if api is not None and image_ops.forget(api, path):
        invalidate_cache()
    return jsonify({"ok": True, "path": path, "deleted": how})


@bp.post("/api/forget")
def forget_file():
    """Remove a file from the search index; the file and the duplicate results stay."""
    path = (request.get_json(force=True) or {}).get("path", "")
    if not (_within_root(path) and _known_path(path)):
        abort(400, "Path is not a file from the current scan results.")
    api = _index()
    if api is None or not image_ops.forget(api, path):
        abort(400, "That file isn't in the search index.")
    invalidate_cache()
    return jsonify({"ok": True, "path": path})


@bp.post("/api/reveal")
def reveal():
    path = request.get_json(force=True).get("path", "")
    if not (_within_root(path) and os.path.exists(path)):
        abort(400)
    subprocess.run(["open", "-R", path])
    return jsonify({"ok": True})


@bp.get("/api/thumb")
def thumb():
    path = request.args.get("path", "")
    if not (_within_root(path) and core.is_image(path) and os.path.isfile(path)):
        abort(404)
    try:
        img = image_sim._load(path, max_side=360)
    except Exception:
        abort(415)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=80)
    buf.seek(0)
    resp = send_file(buf, mimetype="image/jpeg")
    resp.headers["Cache-Control"] = "private, max-age=300"
    return resp


@bp.errorhandler(400)
@bp.errorhandler(403)
@bp.errorhandler(409)
@bp.errorhandler(500)
def _json_error(e):
    return jsonify({"error": getattr(e, "description", str(e))}), e.code
