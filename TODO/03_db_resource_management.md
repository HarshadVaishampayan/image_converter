# 03 — DB & Resource Management (use context managers)

Goal

- Ensure all DB and file operations use context managers (`with` blocks) to avoid leaks and locking issues.

Acceptance criteria

- Replace `conn = sqlite3.connect(...); conn.close()` patterns with `with sqlite3.connect(...) as conn:` where appropriate.
- Use `with open(path, 'rb')` for files.
- Wrap any persistent client usage (e.g., ChromaDB client) with safe init/cleanup if required.

Notes

- Where transactions are needed, rely on `with conn:` commit behavior or explicit `conn.commit()` inside the context.
- Add unit tests that simulate exceptions to ensure resources are properly closed.
