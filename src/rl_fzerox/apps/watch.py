# src/rl_fzerox/apps/watch.py
from __future__ import annotations

from collections.abc import Sequence

from rl_fzerox.apps.watch_cli import parse_args, resolve_watch_app_config
from rl_fzerox.ui.watch import run_viewer


def main(argv: Sequence[str] | None = None) -> None:
    """Load the watch config and launch the viewer."""

    args = parse_args(argv)
    try:
        config = resolve_watch_app_config(
            config_path=args.config_path,
            policy_run_dir=args.policy_run_dir,
            policy_artifact=args.policy_artifact,
            manager_db_path=args.manager_db_path,
            managed_run_id=args.managed_run_id,
            overrides=args.overrides,
        )
    except (RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    run_viewer(config)


if __name__ == "__main__":
    main()
