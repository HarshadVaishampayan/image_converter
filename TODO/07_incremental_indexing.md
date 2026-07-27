# 07 — Implement Incremental Indexing (avoid full rebuilds)

Goal

- Avoid full FAISS/ChromaDB rebuilds on every insert. Implement incremental additions to the index.

Acceptance criteria

- `insert_or_update` upserts vectors incrementally where supported by the backend.
- A migration/check utility exists to validate index consistency.

Notes

- ChromaDB supports upsert; use `collection.upsert` for incremental updates (already used in `v2/persistence.py`).
- For FAISS (v1), implement `index.add()` per new vectors and persist the index map.
- Add a background worker or batched flush if many inserts happen concurrently.
