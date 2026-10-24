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
import time
import uuid

from flask import Blueprint, abort, jsonify, render_template, request, send_file

from . import core, image_sim

bp = Blueprint("duplicates", __name__, url_prefix="/duplicates")

# ------------------------------------------------------------------ state ----
state_lock = threading.Lock()
state = {"root": None, "files": None, "pairs": {}}  # pairs: key -> pair dict
jobs = {}
job_lock = threading.Lock()  # one scan at a time


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


class Job:
    def __init__(self, kind):
        self.id = uuid.uuid4().hex[:12]
        self.kind = kind
        self.status = "running"
        self.done = 0
        self.total = None
        self.message = "Starting…"
        self.error = None
        self.cancelled = False
        self.started = time.time()
        self.result_count = 0

    def progress(self, done, total, message):
        self.done, self.total, self.message = done, total, message

    def to_json(self):
        return {"id": self.id, "kind": self.kind, "status": self.status,
                "done": self.done, "total": self.total, "message": self.message,
                "error": self.error, "elapsed": round(time.time() - self.started, 1),
                "result_count": self.result_count}


def _start_job(kind, fn):
    if not job_lock.acquire(blocking=False):
        abort(409, "Another scan is already running.")
    job = Job(kind)
    jobs[job.id] = job

    def run():
        try:
            job.result_count = fn(job) or 0
            job.status = "done"
            job.message = f"Done — {job.result_count:,} result(s)."
        except core.Cancelled:
            job.status = "cancelled"
            job.message = "Cancelled."
        except Exception as e:  # surface to UI
            job.status = "error"
            job.error = f"{type(e).__name__}: {e}"
            job.message = "Failed."
            import traceback
            traceback.print_exc()
        finally:
            job_lock.release()

    threading.Thread(target=run, daemon=True).start()
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


@bp.post("/api/pick-folder")
def pick_folder():
    """Open the native macOS folder chooser."""
    start = request.json.get("start") if request.is_json else None
    script = 'POSIX path of (choose folder with prompt "Choose a folder to scan for duplicates"'
    if start and os.path.isdir(start):
        script += ' default location (POSIX file "%s")' % start.replace('"', '')
    script += ")"
    try:
        out = subprocess.run(["osascript", "-e", script], capture_output=True,
                             text=True, timeout=600)
    except Exception as e:
        return jsonify({"path": None, "error": str(e)})
    if out.returncode != 0:
        return jsonify({"path": None})  # user cancelled
    path = out.stdout.strip()
    if len(path) > 1:
        path = path.rstrip("/")
    return jsonify({"path": path})


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
    job = jobs.get(job_id) or abort(404)
    return jsonify(job.to_json())


@bp.post("/api/job/<job_id>/cancel")
def job_cancel(job_id):
    job = jobs.get(job_id) or abort(404)
    job.cancelled = True
    return jsonify(job.to_json())


@bp.get("/api/pairs")
def get_pairs():
    with state_lock:
        pairs = list(state["pairs"].values())
    return jsonify({"root": state["root"], "pairs": pairs})


def _move_to_trash(path):
    """Move to the macOS Trash via NSFileManager (supports Finder 'Put Back')."""
    js = ("function run(argv){ObjC.import('Foundation');"
          "var err=$();"
          "var ok=$.NSFileManager.defaultManager.trashItemAtURLResultingItemURLError("
          "$.NSURL.fileURLWithPath(argv[0]), null, err);"
          "if(!ok){throw new Error(ObjC.unwrap(err.localizedDescription))}"
          "return 'ok'}")
    out = subprocess.run(["osascript", "-l", "JavaScript", "-e", js, path],
                         capture_output=True, text=True, timeout=30)
    if out.returncode != 0 or os.path.exists(path):
        raise OSError(out.stderr.strip() or "Could not move file to Trash")


@bp.post("/api/delete")
def delete_file():
    path = request.get_json(force=True).get("path", "")
    if not (_within_root(path) and _known_path(path) and os.path.isfile(path)):
        abort(400, "Path is not a file from the current scan results.")
    try:
        _move_to_trash(path)
    except OSError as e:
        abort(500, str(e))
    with state_lock:
        state["pairs"] = {k: p for k, p in state["pairs"].items()
                          if path not in (p["a"]["path"], p["b"]["path"])}
        if state["files"] is not None:
            state["files"] = [f for f in state["files"] if f["path"] != path]
    return jsonify({"ok": True, "trashed": path})


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
