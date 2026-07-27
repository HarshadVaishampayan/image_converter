# 04 — Store Metadata as JSON (instead of str)

Goal

- Use `json.dumps` / `json.loads` for structured fields (captions, metadata) stored in persistence.

Acceptance criteria

- `captions` and `metadata` fields are saved as JSON strings in ChromaDB or SQLite columns.
- Reading functions parse JSON and return native Python lists/dicts.
- Backward compatibility: gracefully handle legacy `str(...)` values by attempting `json.loads()` first, then fallback to original behavior.

Notes

- Update `v2/persistence.py` `insert_or_update`, `get_image`, and `list_images` to use JSON.
- Add migration helper script to normalize existing entries if needed.
