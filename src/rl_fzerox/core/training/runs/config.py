# src/rl_fzerox/core/training/runs/config.py
from __future__ import annotations

import shutil
from pathlib import Path

from omegaconf import OmegaConf
from pydantic import ValidationError

from rl_fzerox.core.config.paths import config_root_dir, resolve_config_data_paths
from rl_fzerox.core.config.schema import (
    TrainAppConfig,
    TrainConfig,
    WatchAppConfig,
)
from rl_fzerox.core.config.track_registry import expand_track_registry_metadata
from rl_fzerox.core.domain.training_algorithms import (
    TRAIN_ALGORITHM_AUTO,
    TRAIN_ALGORITHM_MASKABLE_PPO,
)
from rl_fzerox.core.training.runs.baseline_materializer import materialize_run_baselines
from rl_fzerox.core.training.runs.migration import scrub_obsolete_train_config_data
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

    baseline_source_path = (
        None
        if watch_config.env.track_sampling.enabled
        else watch_config.emulator.baseline_state_path
    )
    paths = build_watch_session_paths(
        run_dir=run_dir,
        runtime_dir=watch_config.emulator.runtime_dir,
        baseline_state_path=baseline_source_path,
        session_name=session_name,
    )
    ensure_watch_session_dirs(paths)
    _copy_state_file(
        source=baseline_source_path,
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
    baseline_cache_root: Path | None = None,
) -> TrainAppConfig:
    """Rewrite one train config to use run-local runtime and baseline files."""

    materialized_config = materialize_run_baselines(
        config,
        run_paths=run_paths,
        cache_root=baseline_cache_root,
    )
    return materialized_config.model_copy(
        update={
            # Persist the concrete training algorithm so future watch/inference
            # does not need to guess what one historical `auto` meant.
            "train": materialized_config.train.model_copy(
                update={
                    "algorithm": (
                        TRAIN_ALGORITHM_MASKABLE_PPO
                        if materialized_config.train.algorithm == TRAIN_ALGORITHM_AUTO
                        else materialized_config.train.algorithm
                    )
                }
            ),
        }
    )


def save_train_run_config(*, config: TrainAppConfig, run_dir: Path) -> Path:
    """Persist one resolved train config snapshot next to a training run."""

    config_path = run_dir / RUN_LAYOUT.config_filename
    data = _train_config_snapshot_data(config)
    OmegaConf.save(config=OmegaConf.create(data), f=str(config_path))
    return config_path


def load_train_run_config(run_dir: Path) -> TrainAppConfig:
    """Load one previously saved resolved train config snapshot."""

    config_path = resolve_train_run_config_path(run_dir)
    loaded = _load_train_config_mapping(config_path)
    expand_track_registry_metadata(
        loaded,
        config_root=config_root_dir().resolve(),
    )
    # V4 LEGACY SHIM: old local run manifests may contain fields that were
    # deliberately removed from the canonical schema. Keep this isolated so it
    # can be deleted with the migration tool when v4 checkpoints are obsolete.
    scrub_obsolete_train_config_data(loaded)
    _resolve_train_config_paths(loaded, config_dir=config_path.parent)
    return TrainAppConfig.model_validate(loaded)


def load_train_run_config_for_watch(run_dir: Path) -> TrainAppConfig:
    """Load a saved train config for watch and explain stale-manifest failures."""

    try:
        return load_train_run_config(run_dir)
    except ValidationError as exc:
        resolved_run_dir = run_dir.expanduser().resolve()
        raise RuntimeError(
            "Saved train config is not compatible with the current schema: "
            f"{resolved_run_dir}. Run "
            "`python -m rl_fzerox.apps.scrub_train_config --run-dir "
            f"{resolved_run_dir} --in-place` to rewrite a stale local manifest, "
            "or restart the run with the current config schema."
        ) from exc


def load_train_run_train_config(run_dir: Path) -> TrainConfig:
    """Load only the train section from a saved run config."""

    config_path = resolve_train_run_config_path(run_dir)
    loaded = _load_train_config_mapping(config_path)
    train_data = loaded.get("train")
    if not isinstance(train_data, dict):
        raise ValueError(f"Train run config is missing a train mapping: {config_path}")

    train_section: dict[str, object] = {}
    for key, value in train_data.items():
        if not isinstance(key, str):
            raise ValueError(f"Train run config train keys must be strings: {config_path}")
        train_section[key] = value
    _resolve_train_config_paths({"train": train_section}, config_dir=config_path.parent)
    return TrainConfig.model_validate(train_section)


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


def _train_config_snapshot_data(config: TrainAppConfig) -> dict[str, object]:
    data = {
        str(key): value for key, value in config.model_dump(mode="json", exclude_none=True).items()
    }
    _prefer_action_branch_snapshot(data)
    _prefer_observation_component_snapshot(data)
    return data


def _prefer_action_branch_snapshot(config_data: dict[str, object]) -> None:
    env_data = config_data.get("env")
    if not isinstance(env_data, dict):
        return

    action_data = env_data.get("action")
    if not isinstance(action_data, dict):
        return

    branches_data = action_data.get("branches")
    if branches_data is None:
        return

    # Runtime bridge: branch configs compile to adapter-era fields internally.
    # Do not persist those generated fields in fresh run manifests; keeping the
    # branch declaration as the only saved source makes this bridge removable.
    action_data.clear()
    action_data["branches"] = branches_data


def _prefer_observation_component_snapshot(config_data: dict[str, object]) -> None:
    env_data = config_data.get("env")
    if not isinstance(env_data, dict):
        return

    observation_data = env_data.get("observation")
    if not isinstance(observation_data, dict):
        return

    if observation_data.get("state_components") is None:
        return

    for key in (
        "state_profile",
        "course_context",
        "ground_effect_context",
        "action_history_len",
        "action_history_controls",
    ):
        observation_data.pop(key, None)


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
    if train_config.env.track_sampling.enabled:
        merged_emulator_config = merged_emulator_config.model_copy(
            update={"baseline_state_path": None}
        )
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
        not train_config.env.track_sampling.enabled
        and merged_emulator_config.baseline_state_path is None
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
