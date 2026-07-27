# 02 — Replace `print()` with `logging`

Goal

- Replace ad-hoc `print()` calls with Python `logging`.
- Configure a single `logging` setup in the CLI entrypoint and the Flask app.

Acceptance criteria

- All modules use `logger = logging.getLogger(__name__)`.
- `v2/image_cli.py` configures root logging and propagates settings.
- No `print()` calls remain for diagnostic messages (except in very small scripts/tests).

Notes

- Use `logger.info`, `logger.debug`, `logger.warning`, and `logger.exception` appropriately.
- Keep `print()` for human-facing short CLI output if deliberate, but prefer `click.echo()` in CLI commands.
