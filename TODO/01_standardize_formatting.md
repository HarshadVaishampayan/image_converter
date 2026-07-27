# 01 — Standardize Formatting & Linting

Goal

- Normalize code formatting across the repo (convert tabs -> 4-space indentation).
- Adopt automated formatting and linting (Black + Ruff/Flake8).
- Add `pre-commit` hooks to enforce formatting on commit.

Acceptance criteria

- `black .` formats repository without syntax changes.
- `ruff check .` passes or reports a small set of fixable warnings.
- A `pre-commit` config is present and `pre-commit install` runs successfully.

Suggested commands

```bash
pip install black ruff pre-commit
black .
ruff check .
pre-commit install
pre-commit run --all-files
```
