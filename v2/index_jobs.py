"""On-disk records of indexing runs, so status, failures and resume survive restarts.

One JSON file per run in v2/logs/index_jobs/. A run that is still "running" when
the app starts again was interrupted (the process died), and is marked so.
"""
import json
import os
import threading
import time
import uuid

from app_logging import LOG_DIR

JOBS_DIR = os.path.join(LOG_DIR, "index_jobs")
MAX_ISSUES = 5000  # cap per list so one broken folder can't bloat a record

_lock = threading.Lock()

# statuses
RUNNING, DONE, CANCELLED, FAILED, INTERRUPTED = "running", "done", "cancelled", "failed", "interrupted"


def _path(record_id):
    return os.path.join(JOBS_DIR, f"{record_id}.json")


def save(rec):
    rec["updated"] = time.time()
    os.makedirs(JOBS_DIR, exist_ok=True)
    tmp = _path(rec["id"]) + ".tmp"
    with _lock:
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(rec, fh, indent=1)
        os.replace(tmp, _path(rec["id"]))  # atomic: never a half-written record


def create(folder, recursive, caption_model, total, origin="new", parent=None):
    rec = {
        "id": time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6],
        "folder": folder,
        "recursive": recursive,
        "caption_model": caption_model,
        "origin": origin,          # "new" | "resume" | "retry"
        "parent": parent,          # record this one resumes/retries
        "status": RUNNING,
        "created": time.time(),
        "total": total,
        "indexed": 0,
        "failed": [],              # [{"path", "reason", "time"}]
        "skipped": [],             # same shape: unreadable output, empty description…
        "error": None,
    }
    save(rec)
    return rec


def add_issue(rec, kind, path, reason):
    items = rec["failed" if kind == "failed" else "skipped"]
    if len(items) < MAX_ISSUES:
        items.append({"path": path, "reason": reason, "time": time.time()})


def get(record_id):
    if not record_id or "/" in record_id or "\\" in record_id:
        return None
    try:
        with open(_path(record_id), encoding="utf-8") as fh:
            return json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def recent(limit=10):
    try:
        names = sorted((n for n in os.listdir(JOBS_DIR) if n.endswith(".json")), reverse=True)
    except FileNotFoundError:
        return []
    out = []
    for name in names[:limit]:
        rec = get(name[:-5])
        if rec:
            out.append(rec)
    return out


def mark_interrupted():
    """Call once at startup: any run still 'running' died with the previous process."""
    count = 0
    for rec in recent(limit=1000):
        if rec.get("status") == RUNNING:
            rec["status"] = INTERRUPTED
            rec["error"] = rec.get("error") or "The app stopped while this run was in progress."
            save(rec)
            count += 1
    return count


def summary(rec, max_items=200):
    """Record trimmed for the UI: counts plus the first issues."""
    attempted = rec["indexed"] + len(rec["failed"]) + len(rec["skipped"])
    return {
        **{k: rec[k] for k in ("id", "folder", "recursive", "caption_model", "origin", "status",
                               "created", "updated", "total", "indexed", "error")},
        "failed_count": len(rec["failed"]),
        "skipped_count": len(rec["skipped"]),
        "not_attempted": max(0, rec["total"] - attempted),
        "failed": rec["failed"][:max_items],
        "skipped": rec["skipped"][:max_items],
    }
