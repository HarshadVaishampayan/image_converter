# 08 — Consolidate Shared Core Between `v1` and `v2`

Goal

- Reduce duplication between `v1` and `v2` by extracting common utilities into `core/` or `lib/`.

Acceptance criteria

- Common modules (embedding wrappers, utility IO, and search helpers) are available under `core/` and both v1/v2 import them.
- No behavioral changes in `v2`; `v1` remains for reference but imports from `core/` where safe.

Notes

- Prioritize extracting: `embeddings` logic, helper functions for file scanning, and common DB helpers.
- Keep `v1` intact until tests confirm parity.
