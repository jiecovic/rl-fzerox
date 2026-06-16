# src/rl_fzerox/apps/career_mode_cli/args.py
from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from rl_fzerox.core.manager import default_manager_db_path


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the F-Zero X Career Mode viewer for one managed save game.",
        allow_abbrev=False,
    )
    parser.add_argument(
        "overrides",
        nargs=argparse.REMAINDER,
        help="Viewer overrides. Use `-- key=value` to separate them from CLI flags.",
    )
    parser.add_argument(
        "--manager-db-path",
        dest="manager_db_path",
        type=Path,
        default=default_manager_db_path(),
        help="Manager SQLite database path.",
    )
    parser.add_argument(
        "--save-game-id",
        dest="save_game_id",
        required=True,
        help="Managed save-game id to run.",
    )
    parser.add_argument(
        "--save-attempt-id",
        dest="save_attempt_id",
        default=None,
        help="Managed save-attempt id to run.",
    )
    parser.add_argument(
        "--viewer-lease-id",
        dest="viewer_lease_id",
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--attempt-seed",
        dest="attempt_seed",
        type=int,
        default=None,
        help="Seed used only for stochastic policy playback.",
    )
    parser.add_argument(
        "--policy-mode",
        choices=("deterministic", "stochastic"),
        default="deterministic",
        help="Initial policy playback mode; the viewer hotkey can still toggle it.",
    )
    parser.add_argument(
        "--single-target",
        action="store_true",
        help="Keep replaying the selected save target instead of advancing the unlock path.",
    )
    parser.add_argument(
        "--perfect-run",
        action="store_true",
        help="For a selected target, restart the cup attempt after the first crash or retire.",
    )
    parser.add_argument(
        "--discard-failed-recordings",
        action="store_true",
        help="Delete failed target-segment recordings instead of finalizing MP4s.",
    )
    parser.add_argument(
        "--target-clear-goal",
        type=int,
        default=0,
        help=(
            "Stop selected-target replay after this many successful cup clears. "
            "Use 0 to repeat until stopped."
        ),
    )
    args = parser.parse_args(argv)
    if args.attempt_seed is not None and not (0 <= args.attempt_seed <= 0xFFFFFFFF):
        parser.error("--attempt-seed must be between 0 and 4294967295")
    if args.target_clear_goal < 0:
        parser.error("--target-clear-goal must be non-negative")
    if (
        args.perfect_run or args.target_clear_goal > 0 or args.discard_failed_recordings
    ) and not args.single_target:
        parser.error("target fishing options require --single-target")
    return args
