# src/rl_fzerox/core/training/session/artifacts.py
from __future__ import annotations

import json
import os
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

from rl_fzerox.core.config.schema import TrainAppConfig
from rl_fzerox.core.training.runs import RunPaths, materialize_train_run_config


@dataclass(frozen=True)
class PolicyArtifactMetadata:
    """Small sidecar metadata persisted next to one saved policy artifact."""

    curriculum_stage_index: int | None
    curriculum_stage_name: str | None


def resolve_train_run_config(
    *,
    config: TrainAppConfig,
    run_paths: RunPaths,
) -> TrainAppConfig:
    """Resolve one train config snapshot with a run-local runtime root."""

    return materialize_train_run_config(config, run_paths=run_paths)


def validate_training_baseline_state(config: TrainAppConfig) -> None:
    """Fail clearly when a configured local training baseline is missing."""

    baseline_state_path = config.emulator.baseline_state_path
    if baseline_state_path is None:
        return
    if baseline_state_path.exists():
        return
    raise RuntimeError(
        "Configured training baseline state does not exist: "
        f"{baseline_state_path}. Create it from watch with "
        "`emulator.baseline_state_path` set and press `K` at race start."
    )


def save_latest_artifacts(
    model,
    run_paths: RunPaths,
    *,
    policy_metadata: PolicyArtifactMetadata | None = None,
) -> None:
    save_artifacts_atomically(
        model=model,
        model_path=run_paths.latest_model_path,
        policy_path=run_paths.latest_policy_path,
        policy_metadata=policy_metadata,
    )


def save_artifacts_atomically(
    *,
    model,
    model_path: Path,
    policy_path: Path,
    policy_metadata: PolicyArtifactMetadata | None = None,
) -> None:
    atomic_save_artifact(model.save, model_path)
    atomic_save_artifact(model.policy.save, policy_path)
    if policy_metadata is not None:
        _atomic_write_json(_policy_metadata_path(policy_path), asdict(policy_metadata))


def atomic_save_artifact(save_fn, target_path: Path) -> None:
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

    metadata_path = _policy_metadata_path(policy_path)
    if not metadata_path.is_file():
        return None
    data = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return None
    return PolicyArtifactMetadata(
        curriculum_stage_index=_coerce_optional_int(data.get("curriculum_stage_index")),
        curriculum_stage_name=_coerce_optional_str(data.get("curriculum_stage_name")),
    )


def current_policy_artifact_metadata(train_env) -> PolicyArtifactMetadata:
    """Read the currently active curriculum stage from the vector env."""

    return PolicyArtifactMetadata(
        curriculum_stage_index=_first_env_attr(train_env, "curriculum_stage_index"),
        curriculum_stage_name=_first_env_attr(train_env, "curriculum_stage_name"),
    )


def _first_env_attr(train_env, attr_name: str):
    values = train_env.get_attr(attr_name)
    if not values:
        return None
    return values[0]


def _policy_metadata_path(policy_path: Path) -> Path:
    return policy_path.with_name(f"{policy_path.stem}.metadata.json")


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


def cleanup_failed_run(run_paths: RunPaths, model: object | None) -> None:
    if not run_paths.run_dir.exists():
        return

    num_timesteps = getattr(model, "num_timesteps", None) if model is not None else None
    if num_timesteps not in (None, 0):
        return

    shutil.rmtree(run_paths.run_dir, ignore_errors=True)
