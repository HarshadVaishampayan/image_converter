"""Forget-from-index / delete-from-source endpoints, against a throwaway index.

The real Trash is never touched: image_ops.move_to_trash is monkeypatched.
"""
import os
import tempfile

_TMP = tempfile.mkdtemp(prefix="ic-test-")
os.environ["IMAGE_CONVERTER_DB"] = os.path.join(_TMP, "db")        # before importing the app
os.environ["IMAGE_CONVERTER_LOG_DIR"] = os.path.join(_TMP, "logs")

import pytest  # noqa: E402

import gui  # noqa: E402
import image_ops  # noqa: E402

HDR = {"X-Requested-By": "image-converter"}


@pytest.fixture
def client():
    gui.app.config["TESTING"] = True
    return gui.app.test_client()


@pytest.fixture
def photo(tmp_path):
    """A real file on disk with an image entry + a description chunk in the index."""
    path = str(tmp_path / "rose.jpg")
    with open(path, "wb") as fh:
        fh.write(b"not really a jpeg")
    col = gui.api_siglip.collection
    col.upsert(ids=["img-t1", "desc-t1-0"], embeddings=[[1.0, 0, 0, 0], [0, 1.0, 0, 0]],
               metadatas=[{"image_id": "t1", "file_path": path, "type": "image"},
                          {"image_id": "t1", "file_path": path, "type": "description_chunk"}],
               documents=["a rose", "a rose"])
    yield path
    col.delete(where={"file_path": path})


def _entries(path):
    return len(gui.api_siglip.collection.get(where={"file_path": path}, include=[])["ids"])


def test_requires_anti_csrf_header(client, photo):
    assert client.post("/api/images/forget", json={"path": photo}).status_code == 403
    assert client.post("/api/images/delete", json={"path": photo}).status_code == 403
    assert _entries(photo) == 2 and os.path.exists(photo)


def test_forget_removes_all_entries_but_keeps_file(client, photo):
    r = client.post("/api/images/forget", json={"path": photo}, headers=HDR)
    assert r.status_code == 200 and r.json["entries_removed"] == 2
    assert _entries(photo) == 0 and os.path.exists(photo)
    # forgetting again: not in the index any more
    assert client.post("/api/images/forget", json={"path": photo}, headers=HDR).status_code == 404


def test_delete_moves_to_trash_and_forgets(client, photo, monkeypatch):
    trashed = []
    monkeypatch.setattr(image_ops, "move_to_trash", lambda p: (trashed.append(p), os.remove(p)))
    r = client.post("/api/images/delete", json={"path": photo}, headers=HDR)
    assert r.status_code == 200 and r.json["deleted"] == "trash"
    assert trashed == [photo] and not os.path.exists(photo) and _entries(photo) == 0


def test_no_trash_needs_explicit_permanent_confirmation(client, photo, monkeypatch):
    def no_trash(p):
        raise image_ops.TrashUnavailable("volume has no Trash")
    monkeypatch.setattr(image_ops, "move_to_trash", no_trash)

    r = client.post("/api/images/delete", json={"path": photo}, headers=HDR)
    assert r.status_code == 409 and r.json["needs_permanent"] is True
    assert os.path.exists(photo) and _entries(photo) == 2  # nothing happened yet

    r = client.post("/api/images/delete", json={"path": photo, "permanent": True}, headers=HDR)
    assert r.status_code == 200 and r.json["deleted"] == "permanent"
    assert not os.path.exists(photo) and _entries(photo) == 0


def test_refuses_files_that_are_not_indexed(client, tmp_path, monkeypatch):
    other = tmp_path / "keep.txt"
    other.write_text("precious")
    monkeypatch.setattr(image_ops, "move_to_trash", lambda p: pytest.fail("must not delete"))
    for path in (str(other), "/etc/hosts", ""):
        assert client.post("/api/images/delete", json={"path": path, "permanent": True}, headers=HDR).status_code == 404
    assert other.read_text() == "precious"


def test_indexed_but_missing_on_disk(client, photo, monkeypatch):
    os.remove(photo)
    monkeypatch.setattr(image_ops, "move_to_trash", lambda p: pytest.fail("must not delete"))
    r = client.post("/api/images/delete", json={"path": photo}, headers=HDR)
    assert r.status_code == 404 and _entries(photo) == 2  # still forgettable separately
