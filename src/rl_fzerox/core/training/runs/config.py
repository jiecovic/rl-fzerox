# src/rl_fzerox/core/training/runs/config.py
from __future__ import annotations

import shutil
from pathlib import Path

from omegaconf import OmegaConf

from rl_fzerox.core.config.paths import resolve_config_data_paths
from rl_fzerox.core.config.schema import TrainAppConfig, WatchAppConfig
from rl_fzerox.core.domain.training_algorithms import (
    TRAIN_ALGORITHM_AUTO,
    TRAIN_ALGORITHM_MASKABLE_PPO,
)
from rl_fzerox.core.training.runs.paths import (
    RUN_LAYOUT,
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
            ),
            # Persist the concrete training algorithm so future watch/inference
            # does not need to guess what one historical `auto` meant.
            "train": config.train.model_copy(
                update={
                    "algorithm": (
                        TRAIN_ALGORITHM_MASKABLE_PPO
                        if config.train.algorithm == TRAIN_ALGORITHM_AUTO
                        else config.train.algorithm
                    )
                }
            ),
        }
    )


def save_train_run_config(*, config: TrainAppConfig, run_dir: Path) -> Path:
    """Persist one resolved train config snapshot next to a training run."""

    config_path = run_dir / RUN_LAYOUT.config_filename
    data = config.model_dump(mode="json", exclude_none=True)
    OmegaConf.save(config=OmegaConf.create(data), f=str(config_path))
    return config_path


def load_train_run_config(run_dir: Path) -> TrainAppConfig:
    """Load one previously saved resolved train config snapshot."""

    config_path = resolve_train_run_config_path(run_dir)
    loaded = _load_train_config_mapping(config_path)
    _resolve_train_config_paths(loaded, config_dir=config_path.parent)
    return TrainAppConfig.model_validate(loaded)


def _load_train_config_mapping(config_path: Path) -> dict[str, object]:
    loaded = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    if not isinstance(loaded, dict):
        raise ValueError(f"Train run config must resolve to a mapping: {config_path}")

    normalized: dict[str, object] = {}
    for key, value in loaded.items():
        if not isinstance(key, str):
            raise ValueError(f"Train run config keys must be strings: {config_path}")
        normalized[key] = value
    return normalized


def _resolve_train_config_paths(config_data: dict[str, object], *, config_dir: Path) -> None:
    resolve_config_data_paths(
        config_data,
        config_dir=config_dir,
        path_fields={
            "emulator": (
                "core_path",
                "rom_path",
                "runtime_dir",
                "baseline_state_path",
            ),
            "train": (
                "output_root",
                "init_run_dir",
            ),
        },
    )


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
