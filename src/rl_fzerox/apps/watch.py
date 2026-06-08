# src/rl_fzerox/apps/watch.py
from __future__ import annotations

import os
from collections.abc import Sequence

from rl_fzerox.apps.viewer_runtime import manager_viewer_lease_session
from rl_fzerox.apps.watch_cli import parse_args, resolve_watch_app_config
from rl_fzerox.ui.watch import run_viewer


def main(argv: Sequence[str] | None = None) -> None:
    """Resolve one watch session from a run and launch the viewer."""

    args = parse_args(argv)
    with manager_viewer_lease_session(
        db_path=args.manager_db_path,
        lease_id=args.viewer_lease_id,
    ):
        try:
            config = resolve_watch_app_config(
                policy_run_dir=args.policy_run_dir,
                policy_artifact=args.policy_artifact,
                manager_db_path=args.manager_db_path,
                managed_run_id=args.managed_run_id,
                session_name=_watch_session_name(args.viewer_lease_id),
                overrides=args.overrides,
            )
        except (RuntimeError, ValueError) as exc:
            raise SystemExit(str(exc)) from exc

        run_viewer(config)


def _watch_session_name(viewer_lease_id: str | None) -> str | None:
    if not viewer_lease_id:
        return None
    return f"{viewer_lease_id}:{os.getpid()}"


if __name__ == "__main__":
    main()
