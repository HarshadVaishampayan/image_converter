"""Tests for indexing-run records (resume/retry bookkeeping) and the log tail."""
import json

import pytest

import app_logging
import index_jobs


@pytest.fixture(autouse=True)
def jobs_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(index_jobs, "JOBS_DIR", str(tmp_path / "index_jobs"))
    return tmp_path


def test_create_save_and_reload():
    rec = index_jobs.create("/photos", True, "florence2", total=10)
    rec["indexed"] = 4
    index_jobs.add_issue(rec, "failed", "/photos/a.jpg", "could not read image")
    index_jobs.add_issue(rec, "skipped", "/photos/b.jpg", "empty description")
    index_jobs.save(rec)

    again = index_jobs.get(rec["id"])
    assert again["indexed"] == 4
    assert [i["path"] for i in again["failed"]] == ["/photos/a.jpg"]
    assert [i["path"] for i in again["skipped"]] == ["/photos/b.jpg"]


def test_summary_counts_unreached_images():
    rec = index_jobs.create("/photos", False, "ollama", total=10)
    rec["indexed"] = 6
    index_jobs.add_issue(rec, "failed", "/photos/x.jpg", "boom")
    s = index_jobs.summary(rec)
    assert (s["failed_count"], s["skipped_count"], s["not_attempted"]) == (1, 0, 3)


def test_running_records_become_interrupted_on_startup():
    running = index_jobs.create("/a", True, "florence2", total=5)
    finished = index_jobs.create("/b", True, "florence2", total=5)
    finished["status"] = index_jobs.DONE
    index_jobs.save(finished)

    assert index_jobs.mark_interrupted() == 1
    assert index_jobs.get(running["id"])["status"] == index_jobs.INTERRUPTED
    assert index_jobs.get(finished["id"])["status"] == index_jobs.DONE
    assert index_jobs.mark_interrupted() == 0  # idempotent


def test_recent_is_newest_first():
    ids = [index_jobs.create(f"/f{i}", True, "florence2", total=1)["id"] for i in range(3)]
    assert [r["id"] for r in index_jobs.recent()] == sorted(ids, reverse=True)


def test_issue_lists_are_capped(monkeypatch):
    monkeypatch.setattr(index_jobs, "MAX_ISSUES", 3)
    rec = index_jobs.create("/f", True, "florence2", total=10)
    for i in range(10):
        index_jobs.add_issue(rec, "failed", f"/f/{i}.jpg", "bad")
    assert len(rec["failed"]) == 3


@pytest.mark.parametrize("bad_id", ["../etc/passwd", "a/b", "..\\x", "", None])
def test_get_rejects_path_like_ids(bad_id):
    assert index_jobs.get(bad_id) is None


def test_records_are_valid_json_on_disk(jobs_dir):
    rec = index_jobs.create("/f", True, "florence2", total=1)
    with open(jobs_dir / "index_jobs" / f"{rec['id']}.json") as fh:
        assert json.load(fh)["folder"] == "/f"


def test_log_tail(tmp_path, monkeypatch):
    log = tmp_path / "app.log"
    log.write_text("\n".join(f"line {i}" for i in range(1000)))
    monkeypatch.setattr(app_logging, "LOG_FILE", str(log))
    assert app_logging.tail(3).splitlines() == ["line 997", "line 998", "line 999"]
    monkeypatch.setattr(app_logging, "LOG_FILE", str(tmp_path / "missing.log"))
    assert app_logging.tail() == ""
