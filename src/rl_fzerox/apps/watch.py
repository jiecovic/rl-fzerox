# src/rl_fzerox/apps/watch.py
from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from rl_fzerox.core.config import load_watch_app_config
from rl_fzerox.core.training.runs import (
    apply_train_run_to_watch_config,
    load_train_run_config,
)
from rl_fzerox.ui.viewer import run_viewer


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the watch app."""

    parser = argparse.ArgumentParser(
        description="Watch the F-Zero X environment from a Hydra-composed YAML config.",
        allow_abbrev=False,
    )
    parser.add_argument(
        "-c",
        "--config",
        "--config-file",
        dest="config_path",
        required=True,
        type=Path,
        help="Path to a watch config YAML file.",
    )
    parser.add_argument(
        "overrides",
        nargs=argparse.REMAINDER,
        help="Hydra overrides. Use `-- key=value` to separate them from CLI flags.",
    )
    parser.add_argument(
        "--run-dir",
        dest="policy_run_dir",
        type=Path,
        default=None,
        help="Optional training run directory. The watch app loads its latest checkpoint.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Load the watch config and launch the viewer."""

    args = parse_args(argv)
    try:
        config = load_watch_app_config(
            args.config_path,
            overrides=_normalize_overrides(args.overrides),
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    policy_run_dir = (
        args.policy_run_dir.expanduser().resolve()
        if args.policy_run_dir is not None
        else config.watch.policy_run_dir
    )
    if policy_run_dir is not None:
        train_config = load_train_run_config(policy_run_dir)
        config = apply_train_run_to_watch_config(
            config,
            run_dir=policy_run_dir,
            train_config=train_config,
        )

    run_viewer(config)


def _normalize_overrides(overrides: Sequence[str]) -> list[str]:
    if not overrides:
        return []
    if overrides[0] == "--":
        return list(overrides[1:])
    return list(overrides)


if __name__ == "__main__":
    main()
