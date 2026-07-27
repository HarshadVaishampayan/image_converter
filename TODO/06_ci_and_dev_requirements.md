# 06 — CI & Development Requirements

Goal

- Add a `requirements-dev.txt` (or `pyproject.toml`) and a minimal GitHub Actions workflow to run linters and tests on PRs.

Acceptance criteria

- `requirements-dev.txt` created with `black`, `ruff`, `mypy`, `pytest`, `pre-commit`.
- `.github/workflows/ci.yml` runs formatting check, lint, and tests.

Suggested `requirements-dev.txt`

```
black
ruff
mypy
pytest
pre-commit
```

Suggested `ci.yml` steps

- Checkout
- Setup Python
- Install `pip install -r requirements-dev.txt`
- Run `black --check .`, `ruff check .`, `mypy v2 --ignore-missing-imports`, `pytest -q`

Notes

- Keep CI fast by focusing checks on the `v2` directory.
