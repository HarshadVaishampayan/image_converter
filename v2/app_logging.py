"""Logging setup shared by the web app and the CLI.

Everything goes to the console and to a rotating file, v2/logs/image_converter.log
(5 MB x 5 files), so a failed run can be diagnosed after the process is gone.
"""
import logging
import os
from logging.handlers import RotatingFileHandler

LOG_DIR = os.environ.get("IMAGE_CONVERTER_LOG_DIR") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
LOG_FILE = os.path.join(LOG_DIR, "image_converter.log")
_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"

_configured = False


def setup(level=logging.INFO):
    """Idempotently attach console + rotating-file handlers to the root logger."""
    global _configured
    if _configured:
        return
    os.makedirs(LOG_DIR, exist_ok=True)
    root = logging.getLogger()
    root.setLevel(level)
    fmt = logging.Formatter(_FORMAT)

    file_handler = RotatingFileHandler(LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=5, encoding="utf-8")
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)

    console = logging.StreamHandler()
    console.setFormatter(fmt)
    root.addHandler(console)

    # Third-party chatter stays out of our log unless it's a warning or worse
    for noisy in ("werkzeug", "httpx", "httpx2", "httpcore", "huggingface_hub", "urllib3",
                  "chromadb", "transformers", "sentence_transformers", "filelock", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    _configured = True


def tail(lines=300):
    """Last `lines` lines of the current log file (empty string if none yet)."""
    try:
        with open(LOG_FILE, "rb") as fh:
            fh.seek(0, os.SEEK_END)
            size = fh.tell()
            block = min(size, max(lines, 1) * 400)
            fh.seek(size - block)
            data = fh.read().decode("utf-8", errors="replace")
    except FileNotFoundError:
        return ""
    return "\n".join(data.splitlines()[-lines:])
