# src/rl_fzerox/core/manager/projection/launches.py
"""Build full training app configs for fresh runs, resumes, and forks."""

from __future__ import annotations

from pathlib import Path

from rl_fzerox.core.manager.projection.assembly import (
    effective_train_algorithm,
    train_config_payload,
)
from rl_fzerox.core.manager.run_spec import ManagedRunConfig
from rl_fzerox.core.runtime_spec.paths import project_root_dir
from rl_fzerox.core.runtime_spec.schema import TrainAppConfig
from rl_fzerox.core.runtime_spec.track_registry import expand_track_registry_metadata


def _validated_run_spec(config: ManagedRunConfig) -> ManagedRunConfig:
    """Re-run root validation after any in-memory manager config mutations."""

    return ManagedRunConfig.model_validate(config.model_dump(mode="python"))


def build_managed_train_app_config(
    config: ManagedRunConfig,
    *,
    run_id: str,
    run_dir: Path,
) -> TrainAppConfig:
    """Project one manager-owned run spec into the current training schema."""

    config = _validated_run_spec(config)
    train_data = train_config_payload(
        config,
        run_id=run_id,
        run_dir=run_dir,
    )
    expand_track_registry_metadata(
        train_data,
        config_root=project_root_dir().resolve(),
    )
    return TrainAppConfig.model_validate(train_data)


def build_managed_resume_train_app_config(
    config: ManagedRunConfig,
    *,
    run_id: str,
    run_dir: Path,
    tensorboard_step_offset: int = 0,
) -> TrainAppConfig:
    """Project one manager config into an in-place full-model resume run."""

    train_config = build_managed_train_app_config(
        config,
        run_id=run_id,
        run_dir=run_dir,
    )
    return train_config.model_copy(
        update={
            "train": train_config.train.model_copy(
                update={
                    "continue_run_dir": run_dir,
                    "resume_run_dir": run_dir,
                    "resume_source_algorithm": train_config.train.algorithm,
                    "resume_source_auxiliary_state_enabled": (
                        train_config.policy.auxiliary_state.enabled
                    ),
                    "resume_source_auxiliary_state_head_arch": (
                        train_config.policy.auxiliary_state.head_arch
                    ),
                    "resume_artifact": "latest",
                    "resume_mode": "full_model",
                    "tensorboard_step_offset": tensorboard_step_offset,
                }
            )
        }
    )


def build_managed_fork_train_app_config(
    config: ManagedRunConfig,
    *,
    run_id: str,
    run_dir: Path,
    source_run_dir: Path,
    source_artifact: str,
    source_config: ManagedRunConfig | None = None,
    tensorboard_step_offset: int = 0,
) -> TrainAppConfig:
    """Project one manager config into a child run warm-started from another run."""

    train_config = build_managed_train_app_config(
        config,
        run_id=run_id,
        run_dir=run_dir,
    )
    resume_updates: dict[str, object] = {
        "resume_run_dir": source_run_dir,
        "resume_artifact": source_artifact,
        "resume_mode": "weights_only",
        "tensorboard_step_offset": tensorboard_step_offset,
    }
    if source_config is not None:
        resume_updates.update(
            _managed_resume_source_metadata(source_config),
        )
    return train_config.model_copy(
        update={
            "train": train_config.train.model_copy(
                update=resume_updates,
            )
        }
    )


def _managed_resume_source_metadata(config: ManagedRunConfig) -> dict[str, object]:
    return {
        "resume_source_algorithm": effective_train_algorithm(config),
        "resume_source_auxiliary_state_enabled": config.policy.auxiliary_state_enabled,
        "resume_source_auxiliary_state_head_arch": config.policy.auxiliary_state_head_arch,
    }
