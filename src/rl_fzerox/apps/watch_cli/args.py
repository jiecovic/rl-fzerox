# src/rl_fzerox/apps/watch_cli/args.py
from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path


def require_watch_locator(
    *,
    managed_run_id: str | None,
    policy_run_dir: Path | None,
) -> None:
    """Reject ambiguous or missing watch sources before runtime config resolution."""

    source_count = sum(value is not None for value in (policy_run_dir, managed_run_id))
    if source_count > 1:
        raise SystemExit("--run-dir and --managed-run-id are mutually exclusive")
    if source_count == 0:
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
    parser.add_argument(
        "--viewer-lease-id",
        dest="viewer_lease_id",
        default=None,
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args(argv)
    if (
        args.policy_artifact is not None
        and args.policy_run_dir is None
        and args.managed_run_id is None
    ):
        raise SystemExit("--artifact requires --run-dir or --managed-run-id")
    require_watch_locator(
        managed_run_id=args.managed_run_id,
        policy_run_dir=args.policy_run_dir,
    )
    return args
