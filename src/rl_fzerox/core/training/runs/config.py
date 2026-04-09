# src/rl_fzerox/core/training/runs/config.py
from __future__ import annotations

import shutil
from pathlib import Path

from omegaconf import OmegaConf

from rl_fzerox.core.config.schema import TrainAppConfig, WatchAppConfig
from rl_fzerox.core.training.runs.paths import (
    RUN_CONFIG_FILENAME,
    RunPaths,
    build_watch_session_paths,
    ensure_watch_session_dirs,
    resolve_train_run_config_path,
)


def materialize_watch_session_config(
    watch_config: WatchAppConfig,
    *,
    run_dir: Path | None,
    session_name: str | None = None,
) -> WatchAppConfig:
    """Rewrite one watch config to use isolated per-session scratch paths."""

    paths = build_watch_session_paths(
        run_dir=run_dir,
        runtime_dir=watch_config.emulator.runtime_dir,
        baseline_state_path=watch_config.emulator.baseline_state_path,
        session_name=session_name,
    )
    ensure_watch_session_dirs(paths)
    _copy_state_file(
        source=watch_config.emulator.baseline_state_path,
        destination=paths.baseline_state_path,
    )
    return watch_config.model_copy(
        update={
            "emulator": watch_config.emulator.model_copy(
                update={
                    "runtime_dir": paths.runtime_dir,
                    "baseline_state_path": paths.baseline_state_path,
                }
            )
        }
    )


def materialize_train_run_config(
    config: TrainAppConfig,
    *,
    run_paths: RunPaths,
) -> TrainAppConfig:
    """Rewrite one train config to use run-local runtime and baseline files."""

    baseline_state_path = config.emulator.baseline_state_path
    if baseline_state_path is not None:
        _copy_state_file(
            source=baseline_state_path,
            destination=run_paths.baseline_state_path,
        )
    return config.model_copy(
        update={
            "emulator": config.emulator.model_copy(
                update={
                    "runtime_dir": run_paths.runtime_root,
                    "baseline_state_path": (
                        None if baseline_state_path is None else run_paths.baseline_state_path
                    ),
                }
            )
        }
    )


def save_train_run_config(*, config: TrainAppConfig, run_dir: Path) -> Path:
    """Persist one resolved train config snapshot next to a training run."""

    config_path = run_dir / RUN_CONFIG_FILENAME
    data = config.model_dump(mode="json", exclude_none=True)
    OmegaConf.save(config=OmegaConf.create(data), f=str(config_path))
    return config_path


def load_train_run_config(run_dir: Path) -> TrainAppConfig:
    """Load one previously saved resolved train config snapshot."""

    config_path = resolve_train_run_config_path(run_dir)
    loaded = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    if not isinstance(loaded, dict):
        raise ValueError(f"Train run config must resolve to a mapping: {config_path}")
    return TrainAppConfig.model_validate(loaded)


def apply_train_run_to_watch_config(
    watch_config: WatchAppConfig,
    *,
    run_dir: Path,
    train_config: TrainAppConfig,
) -> WatchAppConfig:
    """Inherit emulator/env settings from a training run for policy watch mode."""

    merged_emulator_config = train_config.emulator
    if (
        watch_config.emulator.runtime_dir is not None
        and merged_emulator_config.runtime_dir != watch_config.emulator.runtime_dir
    ):
        merged_emulator_config = merged_emulator_config.model_copy(
            update={
                "runtime_dir": watch_config.emulator.runtime_dir,
            }
        )

    if (
        merged_emulator_config.baseline_state_path is None
        and watch_config.emulator.baseline_state_path is not None
    ):
        merged_emulator_config = merged_emulator_config.model_copy(
            update={
                "baseline_state_path": watch_config.emulator.baseline_state_path,
            }
        )

    return watch_config.model_copy(
        update={
            "seed": train_config.seed,
            "emulator": merged_emulator_config,
            "env": train_config.env,
            "reward": train_config.reward,
            "curriculum": train_config.curriculum,
            "watch": watch_config.watch.model_copy(
                update={"policy_run_dir": run_dir.expanduser().resolve()}
            ),
        }
    )


def _copy_state_file(*, source: Path | None, destination: Path | None) -> None:
    if source is None or destination is None:
        return
    resolved_source = source.expanduser().resolve()
    resolved_destination = destination.expanduser().resolve()
    if resolved_source == resolved_destination:
        return
    if not resolved_source.is_file():
        return
    resolved_destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(resolved_source, resolved_destination)
