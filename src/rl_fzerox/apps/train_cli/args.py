# src/rl_fzerox/apps/train_cli/args.py
from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the training app."""

    parser = argparse.ArgumentParser(
        description="Resume an existing training run in place from its saved manifest.",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--continue-run-dir",
        dest="continue_run_dir",
        type=Path,
        required=True,
        help="Continue an existing run in place, reusing its run directory and tensorboard logs.",
    )
    parser.add_argument(
        "--continue-artifact",
        dest="continue_artifact",
        choices=("latest", "best", "final"),
        default="latest",
        help="Checkpoint artifact to load when --continue-run-dir is used.",
    )
    parser.add_argument(
        "overrides",
        nargs=argparse.REMAINDER,
        help="Train overrides. Use `-- key=value` to separate them from CLI flags.",
    )
    return parser.parse_args(argv)
