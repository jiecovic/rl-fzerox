# src/rl_fzerox/apps/train_cli/cli.py
from __future__ import annotations

from collections.abc import Sequence

from rl_fzerox.apps._cli import normalize_cli_overrides
from rl_fzerox.apps.train_cli.args import parse_args
from rl_fzerox.apps.train_cli.config import continue_saved_run_config
from rl_fzerox.core.training.runner import run_training
from rl_fzerox.core.training.runs import load_train_run_config


def main(argv: Sequence[str] | None = None) -> None:
    """Load the train config and start training."""

    args = parse_args(argv)
    try:
        resolved_continue_run_dir = args.continue_run_dir.expanduser().resolve()
        base_config = load_train_run_config(resolved_continue_run_dir)
        config = continue_saved_run_config(
            base_config,
            continue_run_dir=resolved_continue_run_dir,
            continue_artifact=args.continue_artifact,
            overrides=normalize_cli_overrides(args.overrides),
        )
    except (OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    run_training(config)
