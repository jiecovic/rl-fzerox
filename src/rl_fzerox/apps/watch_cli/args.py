# src/rl_fzerox/apps/watch_cli/args.py
from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path


def require_policy_run_locator(*, policy_run_dir: Path | None, managed_run_id: str | None) -> None:
    """Reject ambiguous or missing policy sources before runtime config resolution."""

    if policy_run_dir is not None and managed_run_id is not None:
        raise SystemExit("--run-dir cannot be combined with --managed-run-id")
    if policy_run_dir is None and managed_run_id is None:
        raise SystemExit("--run-dir or --managed-run-id is required")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the watch app."""

    parser = argparse.ArgumentParser(
        description="Watch the F-Zero X environment from a managed run or run directory.",
        allow_abbrev=False,
    )
    parser.add_argument(
        "overrides",
        nargs=argparse.REMAINDER,
        help="Watch overrides. Use `-- key=value` to separate them from CLI flags.",
    )
    parser.add_argument(
        "--run-dir",
        dest="policy_run_dir",
        type=Path,
        default=None,
        help=(
            "Optional training run directory. The watch app loads its latest saved policy artifact."
        ),
    )
    parser.add_argument(
        "--artifact",
        dest="policy_artifact",
        choices=("latest", "best", "final"),
        default=None,
        help="Which saved policy artifact to load from the run directory.",
    )
    parser.add_argument(
        "--manager-db-path",
        dest="manager_db_path",
        type=Path,
        default=None,
        help="Optional manager SQLite path for manager-owned watch sessions.",
    )
    parser.add_argument(
        "--managed-run-id",
        dest="managed_run_id",
        default=None,
        help="Optional run-manager run id to resolve watch config from SQLite.",
    )
    args = parser.parse_args(argv)
    if (
        args.policy_artifact is not None
        and args.policy_run_dir is None
        and args.managed_run_id is None
    ):
        raise SystemExit("--artifact requires --run-dir or --managed-run-id")
    require_policy_run_locator(
        policy_run_dir=args.policy_run_dir,
        managed_run_id=args.managed_run_id,
    )
    return args
