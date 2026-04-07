# src/rl_fzerox/apps/train.py
from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from rl_fzerox.core.config import load_train_app_config
from rl_fzerox.core.training.runner import run_training


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the PPO training app."""

    parser = argparse.ArgumentParser(
        description="Train an SB3 PPO agent from a Hydra-composed YAML config.",
        allow_abbrev=False,
    )
    parser.add_argument(
        "-c",
        "--config",
        "--config-file",
        dest="config_path",
        required=True,
        type=Path,
        help="Path to a train config YAML file.",
    )
    parser.add_argument(
        "overrides",
        nargs=argparse.REMAINDER,
        help="Hydra overrides. Use `-- key=value` to separate them from CLI flags.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Load the train config and start PPO training."""

    args = parse_args(argv)
    try:
        config = load_train_app_config(
            args.config_path,
            overrides=_normalize_overrides(args.overrides),
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    run_training(config)


def _normalize_overrides(overrides: Sequence[str]) -> list[str]:
    if not overrides:
        return []
    if overrides[0] == "--":
        return list(overrides[1:])
    return list(overrides)


if __name__ == "__main__":
    main()
