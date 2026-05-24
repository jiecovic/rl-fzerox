# src/rl_fzerox/apps/watch.py
from __future__ import annotations

import json
import os
import signal
from collections.abc import Sequence
from contextlib import contextmanager
from pathlib import Path

from rl_fzerox.apps.watch_cli import parse_args, resolve_watch_app_config
from rl_fzerox.ui.watch import run_viewer


def main(argv: Sequence[str] | None = None) -> None:
    """Resolve one watch session from a run and launch the viewer."""

    args = parse_args(argv)
    with _watch_pid_file_session(args.watch_pid_file):
        try:
            config = resolve_watch_app_config(
                policy_run_dir=args.policy_run_dir,
                policy_artifact=args.policy_artifact,
                manager_db_path=args.manager_db_path,
                managed_run_id=args.managed_run_id,
                overrides=args.overrides,
            )
        except (RuntimeError, ValueError) as exc:
            raise SystemExit(str(exc)) from exc

        run_viewer(config)


@contextmanager
def _watch_pid_file_session(pid_path: Path | None):
    cleanup = _watch_pid_file_cleanup(pid_path)
    previous_sigterm = signal.getsignal(signal.SIGTERM)
    previous_sigint = signal.getsignal(signal.SIGINT)

    def _exit_with_cleanup(_signum: int, _frame: object | None) -> None:
        cleanup()
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, _exit_with_cleanup)
    signal.signal(signal.SIGINT, _exit_with_cleanup)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)
        signal.signal(signal.SIGINT, previous_sigint)
        cleanup()


def _watch_pid_file_cleanup(pid_path: Path | None):
    """Unlink one watch pid file only when it still belongs to this process."""

    process_pid = os.getpid()

    def _cleanup() -> None:
        if pid_path is None or not pid_path.is_file():
            return
        try:
            payload = json.loads(pid_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pid_path.unlink(missing_ok=True)
            return
        if payload.get("pid") == process_pid:
            pid_path.unlink(missing_ok=True)

    return _cleanup


if __name__ == "__main__":
    main()
