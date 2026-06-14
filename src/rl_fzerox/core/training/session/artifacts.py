# src/rl_fzerox/core/training/session/artifacts.py
from __future__ import annotations

import json
import os
import shutil
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

from rl_fzerox.core.engine_tuning import EngineTuningRuntimeState
from rl_fzerox.core.engine_tuning.persistence import (
    load_engine_tuning_runtime_state,
    save_engine_tuning_runtime_state,
)
from rl_fzerox.core.runtime_spec.schema import TrainAppConfig
from rl_fzerox.core.training.runs import RUN_LAYOUT, RunPaths, materialize_train_run_config

POLICY_METADATA_SUFFIX = ".metadata.json"


@dataclass(frozen=True)
class PolicyArtifactMetadata:
    """Small sidecar metadata persisted next to one saved policy artifact."""

    curriculum_stage_index: int | None
    curriculum_stage_name: str | None
    num_timesteps: int | None
    lineage_num_timesteps: int | None = None


class SaveArtifactFn(Protocol):
    def __call__(self, path: str) -> object: ...


class ModelSaveable(Protocol):
    def save(self, path: str) -> object: ...


class TrainingEnvAttrReader(Protocol):
    def get_attr(self, attr_name: str) -> list[object]: ...


def resolve_train_run_config(
    *,
    config: TrainAppConfig,
    run_paths: RunPaths,
    startup_reporter: Callable[[str, str], None] | None = None,
) -> TrainAppConfig:
    """Resolve one train manifest with a run-local runtime root."""

    return materialize_train_run_config(
        config,
        run_paths=run_paths,
        startup_reporter=startup_reporter,
    )


def validate_training_baseline_state(config: TrainAppConfig) -> None:
    """Fail clearly when a configured local training baseline is missing."""

    missing_paths = [
        baseline_state_path
        for baseline_state_path in _configured_baseline_state_paths(config)
        if not baseline_state_path.exists()
    ]
    if not missing_paths:
        return
    formatted_paths = ", ".join(str(path) for path in missing_paths)
    raise RuntimeError(
        "Configured training baseline state does not exist: "
        f"{formatted_paths}. Create or materialize the required reset state before training."
    )


def _configured_baseline_state_paths(config: TrainAppConfig) -> tuple[Path, ...]:
    paths: list[Path] = []
    if config.emulator.baseline_state_path is not None:
        paths.append(config.emulator.baseline_state_path)
    if config.track.baseline_state_path is not None:
        paths.append(config.track.baseline_state_path)
    for entry in config.env.track_sampling.entries:
        if entry.baseline_state_path is not None:
            paths.append(entry.baseline_state_path)
    for stage in config.curriculum.stages:
        if stage.track_sampling is None:
            continue
        for entry in stage.track_sampling.entries:
            if entry.baseline_state_path is not None:
                paths.append(entry.baseline_state_path)
    return tuple(dict.fromkeys(paths))


def save_latest_artifacts(
    model: ModelSaveable,
    run_paths: RunPaths,
    *,
    engine_tuning_state: EngineTuningRuntimeState | None = None,
    policy_metadata: PolicyArtifactMetadata | None = None,
) -> None:
    save_artifacts_atomically(
        model=model,
        model_path=run_paths.latest_model_path,
        policy_path=run_paths.latest_policy_path,
        engine_tuning_state=engine_tuning_state,
        policy_metadata=policy_metadata,
    )


def save_recent_checkpoint_artifacts(
    model: ModelSaveable,
    run_paths: RunPaths,
    *,
    engine_tuning_state: EngineTuningRuntimeState | None = None,
    num_timesteps: int,
    policy_metadata: PolicyArtifactMetadata | None = None,
) -> None:
    checkpoint_dir = recent_checkpoint_dir(run_paths, num_timesteps=num_timesteps)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    save_artifacts_atomically(
        model=model,
        model_path=checkpoint_dir / "model.zip",
        policy_path=checkpoint_dir / "policy.zip",
        engine_tuning_state=engine_tuning_state,
        policy_metadata=policy_metadata,
    )


def trim_recent_checkpoint_artifacts(run_paths: RunPaths, *, keep_last: int | None) -> None:
    if keep_last is None:
        return
    checkpoints = list_recent_checkpoint_dirs(run_paths)
    if len(checkpoints) <= keep_last:
        return
    for checkpoint_dir in checkpoints[: len(checkpoints) - keep_last]:
        shutil.rmtree(checkpoint_dir, ignore_errors=True)


def save_artifacts_atomically(
    *,
    model: ModelSaveable,
    model_path: Path,
    policy_path: Path,
    engine_tuning_state: EngineTuningRuntimeState | None = None,
    policy_metadata: PolicyArtifactMetadata | None = None,
) -> None:
    atomic_save_artifact(model.save, model_path)
    atomic_save_artifact(_policy_save_fn(model), policy_path)
    if engine_tuning_state is not None:
        save_engine_tuning_runtime_state(
            engine_tuning_checkpoint_path(policy_path),
            engine_tuning_state,
            model_path=engine_tuning_model_path(policy_path),
        )
    if policy_metadata is not None:
        _atomic_write_json(policy_artifact_metadata_path(policy_path), asdict(policy_metadata))


def atomic_save_artifact(save_fn: SaveArtifactFn, target_path: Path) -> None:
    tmp_path = target_path.with_name(f".{target_path.stem}.tmp{target_path.suffix}")
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        save_fn(str(tmp_path))
        os.replace(tmp_path, target_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def load_policy_artifact_metadata(policy_path: Path) -> PolicyArtifactMetadata | None:
    """Load the optional sidecar metadata for one saved policy artifact."""

    metadata_path = policy_artifact_metadata_path(policy_path)
    if not metadata_path.is_file():
        return None
    data = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return None
    return PolicyArtifactMetadata(
        curriculum_stage_index=_coerce_optional_int(data.get("curriculum_stage_index")),
        curriculum_stage_name=_coerce_optional_str(data.get("curriculum_stage_name")),
        num_timesteps=_coerce_optional_int(data.get("num_timesteps")),
        lineage_num_timesteps=_coerce_optional_int(data.get("lineage_num_timesteps")),
    )


def load_engine_tuning_checkpoint_state(
    policy_path: Path,
) -> EngineTuningRuntimeState | None:
    """Load the optional engine-tuning state stored beside one policy artifact."""

    return load_engine_tuning_runtime_state(
        engine_tuning_checkpoint_path(policy_path),
        model_path=engine_tuning_model_path(policy_path),
    )


def current_policy_artifact_metadata(
    train_env: TrainingEnvAttrReader,
    model: object,
    *,
    lineage_step_offset: int = 0,
) -> PolicyArtifactMetadata:
    """Read the currently active curriculum stage from the vector env."""

    num_timesteps = _current_num_timesteps(model)
    lineage_num_timesteps = None
    if num_timesteps is not None and lineage_step_offset > 0:
        lineage_num_timesteps = int(lineage_step_offset) + num_timesteps
    return PolicyArtifactMetadata(
        curriculum_stage_index=_coerce_optional_int(
            _first_env_attr(train_env, "curriculum_stage_index")
        ),
        curriculum_stage_name=_coerce_optional_str(
            _first_env_attr(train_env, "curriculum_stage_name")
        ),
        num_timesteps=num_timesteps,
        lineage_num_timesteps=lineage_num_timesteps,
    )


def _first_env_attr(train_env: TrainingEnvAttrReader, attr_name: str) -> object | None:
    values = train_env.get_attr(attr_name)
    if not values:
        return None
    return values[0]


def policy_artifact_metadata_path(policy_path: Path) -> Path:
    """Return the sidecar JSON path stored next to one policy checkpoint."""

    return policy_path.with_name(f"{policy_path.stem}{POLICY_METADATA_SUFFIX}")


def engine_tuning_checkpoint_path(policy_path: Path) -> Path:
    """Return the engine-tuning sidecar path stored in one checkpoint directory."""

    return policy_path.with_name(RUN_LAYOUT.engine_tuning_state_filename)


def engine_tuning_model_path(policy_path: Path) -> Path:
    """Return the engine-tuning model path stored in one checkpoint directory."""

    return policy_path.with_name(RUN_LAYOUT.engine_tuning_model_filename)


def recent_checkpoint_dir(run_paths: RunPaths, *, num_timesteps: int) -> Path:
    """Return the directory used for one numbered periodic checkpoint snapshot."""

    return run_paths.checkpoints_dir / f"{num_timesteps:012d}"


def list_recent_checkpoint_dirs(run_paths: RunPaths) -> tuple[Path, ...]:
    """Return numbered checkpoint directories sorted from oldest to newest."""

    if not run_paths.checkpoints_dir.is_dir():
        return ()
    checkpoints = [
        child
        for child in run_paths.checkpoints_dir.iterdir()
        if child.is_dir() and child.name.isdigit()
    ]
    checkpoints.sort(key=lambda path: int(path.name))
    return tuple(checkpoints)


def _atomic_write_json(target_path: Path, data: dict[str, object]) -> None:
    tmp_path = target_path.with_name(f".{target_path.stem}.tmp{target_path.suffix}")
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        tmp_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(tmp_path, target_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _coerce_optional_int(value: object) -> int | None:
    return value if isinstance(value, int) else None


def _coerce_optional_str(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _current_num_timesteps(model: object) -> int | None:
    value = getattr(model, "num_timesteps", None)
    return value if isinstance(value, int) else None


def _policy_save_fn(model: object) -> SaveArtifactFn:
    """Resolve the policy saver from dynamic SB3-style model objects."""

    policy = getattr(model, "policy", None)
    save_fn = getattr(policy, "save", None)
    if not callable(save_fn):
        raise TypeError("Training model policy does not expose a callable save(path) method.")
    return save_fn


def cleanup_failed_run(
    run_paths: RunPaths,
    model: object | None,
    *,
    preserve_run_dir: bool = False,
) -> None:
    if not run_paths.fresh_run:
        return
    if not run_paths.run_dir.exists():
        return
    if preserve_run_dir:
        return

    num_timesteps = getattr(model, "num_timesteps", None) if model is not None else None
    if num_timesteps not in (None, 0):
        return

    shutil.rmtree(run_paths.run_dir, ignore_errors=True)
