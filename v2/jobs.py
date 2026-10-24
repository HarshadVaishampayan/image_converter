"""Tiny background-job runner shared by the web app (indexing, duplicate scans).

Each JobRunner allows one job at a time; the UI polls Job.to_json() for progress.
"""
import threading
import time
import traceback
import uuid


class Cancelled(Exception):
    """Raised by job code when the user cancels."""


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

    def check_cancelled(self):
        if self.cancelled:
            raise Cancelled()

    def to_json(self):
        return {"id": self.id, "kind": self.kind, "status": self.status,
                "done": self.done, "total": self.total, "message": self.message,
                "error": self.error, "elapsed": round(time.time() - self.started, 1),
                "result_count": self.result_count}


class JobRunner:
    """Runs one job at a time in a daemon thread and remembers jobs by id."""

    def __init__(self):
        self._lock = threading.Lock()
        self.jobs = {}
        self.current = None

    def start(self, kind, fn, done_message="Done — {n:,} result(s)."):
        """Start fn(job) -> int in the background. Returns the Job, or None if busy."""
        if not self._lock.acquire(blocking=False):
            return None
        job = Job(kind)
        self.jobs[job.id] = job
        self.current = job

        def run():
            try:
                job.result_count = fn(job) or 0
                job.status = "done"
                job.message = done_message.format(n=job.result_count)
            except Cancelled:
                job.status = "cancelled"
                job.message = "Cancelled."
            except Exception as e:  # surface to the UI
                job.status = "error"
                job.error = f"{type(e).__name__}: {e}"
                job.message = "Failed."
                traceback.print_exc()
            finally:
                self._lock.release()

        threading.Thread(target=run, daemon=True).start()
        return job

    def get(self, job_id):
        return self.jobs.get(job_id)
