# src/rl_fzerox/apps/watch.py
from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from rl_fzerox.apps._cli import normalize_hydra_overrides
from rl_fzerox.core.config import load_watch_app_config
from rl_fzerox.core.config.schema import TrainAppConfig, WatchAppConfig, WatchConfig
from rl_fzerox.core.training.runs import (
    apply_train_run_to_watch_config,
    load_train_run_config,
    materialize_watch_session_config,
)
from rl_fzerox.ui.watch import run_viewer


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
        type=Path,
        default=None,
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
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Load the watch config and launch the viewer."""

    args = parse_args(argv)
    normalized_overrides = normalize_hydra_overrides(args.overrides)
    cli_run_dir = (
        args.policy_run_dir.expanduser().resolve() if args.policy_run_dir is not None else None
    )
    if args.config_path is None:
        if cli_run_dir is None:
            raise SystemExit("--config is required unless --run-dir is provided")
        if normalized_overrides:
            raise SystemExit("Hydra overrides require --config")
        try:
            train_config = load_train_run_config(cli_run_dir)
        except (FileNotFoundError, ValueError) as exc:
            raise SystemExit(str(exc)) from exc
        config = _default_watch_config_from_train_run(
            train_config,
            run_dir=cli_run_dir,
            artifact=args.policy_artifact or "latest",
        )
    else:
        try:
            config = load_watch_app_config(
                args.config_path,
                overrides=normalized_overrides,
            )
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc

    policy_run_dir = cli_run_dir if cli_run_dir is not None else config.watch.policy_run_dir
    if args.policy_artifact is not None and policy_run_dir is None:
        raise SystemExit("--artifact requires --run-dir or watch.policy_run_dir in the config")
    if policy_run_dir is not None:
        try:
            train_config = load_train_run_config(policy_run_dir)
        except (FileNotFoundError, ValueError) as exc:
            raise SystemExit(str(exc)) from exc
        config = apply_train_run_to_watch_config(
            config,
            run_dir=policy_run_dir,
            train_config=train_config,
        )
        if args.policy_artifact is not None:
            config = config.model_copy(
                update={
                    "watch": config.watch.model_copy(
                        update={"policy_artifact": args.policy_artifact}
                    )
                }
            )

    config = materialize_watch_session_config(
        config,
        run_dir=config.watch.policy_run_dir,
    )
    run_viewer(config)


def _default_watch_config_from_train_run(
    train_config: TrainAppConfig,
    *,
    run_dir: Path,
    artifact: Literal["latest", "best", "final"],
) -> WatchAppConfig:
    """Build one minimal watch config directly from a saved train run."""

    return WatchAppConfig(
        seed=train_config.seed,
        emulator=train_config.emulator,
        env=train_config.env,
        reward=train_config.reward,
        watch=WatchConfig(
            policy_run_dir=run_dir,
            policy_artifact=artifact,
        ),
    )


if __name__ == "__main__":
    main()
