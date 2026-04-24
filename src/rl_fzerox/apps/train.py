# src/rl_fzerox/apps/train.py
from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from rl_fzerox.apps._cli import normalize_hydra_overrides
from rl_fzerox.core.config import load_train_app_config
from rl_fzerox.core.training.runner import run_training
from rl_fzerox.core.training.runs import load_train_run_config


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the training app."""

    parser = argparse.ArgumentParser(
        description="Train an SB3 agent from a Hydra-composed YAML config.",
        allow_abbrev=False,
    )
    parser.add_argument(
        "-c",
        "--config",
        "--config-file",
        dest="config_path",
        type=Path,
        help="Path to a train config YAML file.",
    )
    parser.add_argument(
        "--continue-run-dir",
        dest="continue_run_dir",
        type=Path,
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
        help="Hydra overrides. Use `-- key=value` to separate them from CLI flags.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Load the train config and start training."""

    args = parse_args(argv)
    if args.config_path is None and args.continue_run_dir is None:
        raise SystemExit("--config is required unless --continue-run-dir is provided")

    try:
        if args.config_path is not None:
            overrides = list(normalize_hydra_overrides(args.overrides))
            if args.continue_run_dir is not None:
                resolved_continue_run_dir = args.continue_run_dir.expanduser().resolve()
                overrides.extend(
                    [
                        f"train.continue_run_dir={resolved_continue_run_dir}",
                        f"train.resume_run_dir={resolved_continue_run_dir}",
                        f"train.resume_artifact={args.continue_artifact}",
                        "train.resume_mode=full_model",
                    ]
                )
            config = load_train_app_config(
                args.config_path,
                overrides=overrides,
            )
        else:
            if args.overrides:
                raise SystemExit(
                    "Hydra overrides require --config; saved-run continuation without --config "
                    "reuses the stored run config as-is."
                )
            resolved_continue_run_dir = args.continue_run_dir.expanduser().resolve()
            base_config = load_train_run_config(resolved_continue_run_dir)
            config = base_config.model_copy(
                update={
                    "train": base_config.train.model_copy(
                        update={
                            "continue_run_dir": resolved_continue_run_dir,
                            "resume_run_dir": resolved_continue_run_dir,
                            "resume_artifact": args.continue_artifact,
                            "resume_mode": "full_model",
                        }
                    )
                }
            )
    except (OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    run_training(config)


if __name__ == "__main__":
    main()
