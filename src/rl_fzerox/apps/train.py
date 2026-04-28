# src/rl_fzerox/apps/train.py
from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from omegaconf import OmegaConf

from rl_fzerox.apps._cli import normalize_hydra_overrides
from rl_fzerox.core.config import load_train_app_config
from rl_fzerox.core.config.schema import TrainAppConfig
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
            resolved_continue_run_dir = args.continue_run_dir.expanduser().resolve()
            base_config = load_train_run_config(resolved_continue_run_dir)
            config = _continue_saved_run_config(
                base_config,
                continue_run_dir=resolved_continue_run_dir,
                continue_artifact=args.continue_artifact,
                overrides=normalize_hydra_overrides(args.overrides),
            )
    except (OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    run_training(config)


def _continue_saved_run_config(
    config: TrainAppConfig,
    *,
    continue_run_dir: Path,
    continue_artifact: str,
    overrides: Sequence[str],
) -> TrainAppConfig:
    """Apply optional dotlist overrides to a saved config, then force in-place resume."""

    data = config.model_dump(mode="json", exclude_unset=True)
    if overrides:
        merged = OmegaConf.merge(OmegaConf.create(data), OmegaConf.from_dotlist(list(overrides)))
        loaded = OmegaConf.to_container(merged, resolve=True)
        if not isinstance(loaded, dict):
            raise ValueError("Saved-run overrides must resolve to a mapping")
        data = loaded

    train_data = data.setdefault("train", {})
    if not isinstance(train_data, dict):
        raise ValueError("Saved train config must contain a train mapping")
    train_data.update(
        {
            "continue_run_dir": continue_run_dir,
            "resume_run_dir": continue_run_dir,
            "resume_artifact": continue_artifact,
            "resume_mode": "full_model",
        }
    )
    return TrainAppConfig.model_validate(data)


if __name__ == "__main__":
    main()
