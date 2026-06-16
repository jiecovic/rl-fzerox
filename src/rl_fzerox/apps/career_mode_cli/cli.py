# src/rl_fzerox/apps/career_mode_cli/cli.py
from __future__ import annotations

from collections.abc import Sequence

from rl_fzerox.apps.career_mode_cli.args import parse_args
from rl_fzerox.apps.career_mode_cli.config import resolve_career_mode_config
from rl_fzerox.apps.viewer_runtime import manager_viewer_lease_session
from rl_fzerox.ui.watch import run_viewer
from rl_fzerox.ui.watch.runtime import start_career_mode_worker


def main(argv: Sequence[str] | None = None) -> None:
    """Launch the visible Career Mode runner for one managed save game."""

    args = parse_args(argv)
    db_path = args.manager_db_path.expanduser().resolve()
    with manager_viewer_lease_session(
        db_path=db_path,
        lease_id=args.viewer_lease_id,
    ) as lease_session:
        try:
            config = resolve_career_mode_config(
                db_path=db_path,
                attempt_seed=args.attempt_seed,
                deterministic_policy=args.policy_mode == "deterministic",
                save_attempt_id=args.save_attempt_id,
                save_game_id=args.save_game_id,
                single_target=args.single_target,
                perfect_run=args.perfect_run,
                keep_failed_recordings=not args.discard_failed_recordings,
                target_clear_goal=args.target_clear_goal,
                overrides=args.overrides,
            )
        except (FileNotFoundError, KeyError, RuntimeError, ValueError) as exc:
            raise SystemExit(str(exc)) from exc
        run_viewer(
            config,
            worker_factory=start_career_mode_worker,
            viewer_heartbeat=lease_session.heartbeat,
        )
