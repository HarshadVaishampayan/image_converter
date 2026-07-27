# 05 — Add Type Annotations (mypy)

Goal

- Add type annotations to core modules (`persistence.py`, `model_manager.py`, `core/embeddings.py`, `find_similar.py`) to catch issues early.

Acceptance criteria

- `mypy` runs with no blocking errors for annotated modules (use `--ignore-missing-imports` initially).
- Public functions and methods have docstrings and type signatures.

Suggested commands

```bash
pip install mypy
mypy v2 --ignore-missing-imports
```

Notes

- Start with small modules and incrementally annotate more code.
- Keep runtime behavior unchanged; use `typing.Optional`, `List`, `Dict`, and `TypedDict` where helpful.
