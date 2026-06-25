# src/rl_fzerox/apps/watch_cli/args.py
from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path


def require_run_id(run_id: str | None) -> None:
    """Reject missing watch sources before runtime config resolution."""

    if run_id is None:
        raise SystemExit("--run-id is required")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the watch app."""

    parser = argparse.ArgumentParser(
        description="Watch the F-Zero X environment from a run-manager run.",
        allow_abbrev=False,
    )
    parser.add_argument(
        "overrides",
        nargs=argparse.REMAINDER,
        help="Watch overrides. Use `-- key=value` to separate them from CLI flags.",
    )
    parser.add_argument(
        "--artifact",
        dest="policy_artifact",
        choices=("latest", "best", "final"),
        default=None,
        help="Which saved policy artifact to load from the managed run.",
    )
    parser.add_argument(
        "--manager-db-path",
        dest="manager_db_path",
        type=Path,
        default=None,
        help="Optional manager SQLite path for manager-owned watch sessions.",
    )
    parser.add_argument(
        "--run-id",
        dest="run_id",
        default=None,
        help="Run-manager run id to resolve from SQLite.",
    )
    parser.add_argument(
        "--viewer-lease-id",
        dest="viewer_lease_id",
        default=None,
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args(argv)
    require_run_id(args.run_id)
    return args
