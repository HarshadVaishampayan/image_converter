# 09 — Expand Tests and Use Mocks for Heavy Models

Goal

- Add unit tests for persistence, search, and small helper functions. Mock heavy model loads (Transformers / Ollama) so tests run fast.

Acceptance criteria

- Tests exist for `v2/persistence.py` (insert_or_update, find_similar_images, cleanup) using temporary Chromadb instances or mocking.
- Model-heavy calls are mocked (e.g., `ModelManager.get_embedding_model` returns a lightweight dummy object for tests).
- `pytest` runs quickly in CI (< 2 minutes ideally for unit tests).

Notes

- Use `pytest-mock` or `unittest.mock`.
- For integration tests, have small fixture datasets and a short-lived Chromadb path (temp dir).
