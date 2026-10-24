"""Forget-from-index and delete-from-source, shared by the Search and Duplicates pages.

* forget(api, path): remove every index entry for a file (image, captions,
  description chunks). The file itself is untouched; re-indexing its folder
  brings it back.
* delete_file(path, permanent): move the file to the macOS Trash. Some volumes
  (typically SMB network shares) have no Trash; then TrashUnavailable is raised
  and the caller must ask the user before retrying with permanent=True.
"""
import logging
import os
import subprocess

log = logging.getLogger("image_converter.files")


class TrashUnavailable(OSError):
    """The file can't be moved to the Trash (e.g. it's on a network share)."""


def safe_file(path):
    """True if path is an existing regular file that isn't reached via a symlink or '..'."""
    if not path or not isinstance(path, str):
        return False
    return os.path.realpath(path) == os.path.abspath(path) and os.path.isfile(path)


# ------------------------------------------------------------------ index ----
def indexed_paths(api):
    entries = api.collection.get(where={"type": "image"}, include=["metadatas"])
    return {m.get("file_path") for m in entries.get("metadatas") or []}


def is_indexed(api, path):
    return bool(api.collection.get(where={"file_path": path}, include=[], limit=1)["ids"])


def forget(api, path):
    """Remove every index entry for `path`. Returns how many entries were removed."""
    ids = api.collection.get(where={"file_path": path}, include=[])["ids"]
    if ids:
        api.collection.delete(ids=ids)
        log.info("Forgot %s (%d index entries)", path, len(ids))
    return len(ids)


# ------------------------------------------------------------------- disk ----
_TRASH_JS = (
    "function run(argv){ObjC.import('Foundation');"
    "var err=$();"
    "var ok=$.NSFileManager.defaultManager.trashItemAtURLResultingItemURLError("
    "$.NSURL.fileURLWithPath(argv[0]), null, err);"
    "if(!ok){throw new Error(ObjC.unwrap(err.localizedDescription))}"
    "return 'ok'}"
)


def move_to_trash(path):
    """Move to the macOS Trash via NSFileManager (Finder's 'Put Back' works)."""
    out = subprocess.run(["osascript", "-l", "JavaScript", "-e", _TRASH_JS, path],
                         capture_output=True, text=True, timeout=60)
    if out.returncode != 0 or os.path.exists(path):
        reason = (out.stderr or "").strip().split("Error:")[-1].strip() or "the Trash isn't available here"
        raise TrashUnavailable(reason)


def delete_file(path, permanent=False):
    """Trash `path`; if that's impossible and permanent=True, delete it outright.

    Returns "trash" or "permanent". Raises TrashUnavailable when the Trash can't
    be used and permanent deletion wasn't allowed.
    """
    try:
        move_to_trash(path)
        log.info("Moved to Trash: %s", path)
        return "trash"
    except TrashUnavailable as e:
        if not permanent:
            log.warning("Trash unavailable for %s: %s", path, e)
            raise
        os.remove(path)
        log.warning("PERMANENTLY deleted (no Trash available): %s", path)
        return "permanent"
