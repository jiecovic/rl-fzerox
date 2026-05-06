"""Build full training app configs for fresh runs, resumes, and forks."""

from __future__ import annotations

from pathlib import Path

from rl_fzerox.core.manager.projection.assembly import (
    train_config_payload,
    validate_launch_support,
)
from rl_fzerox.core.manager.run_spec import ManagedRunConfig
from rl_fzerox.core.runtime_spec.paths import project_root_dir
from rl_fzerox.core.runtime_spec.schema import TrainAppConfig
from rl_fzerox.core.runtime_spec.track_registry import expand_track_registry_metadata


def build_managed_train_app_config(
    config: ManagedRunConfig,
    *,
    run_id: str,
    run_dir: Path,
) -> TrainAppConfig:
    """Project one manager-owned run spec into the current training schema."""

    validate_launch_support(config)
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
    tensorboard_step_offset: int = 0,
) -> TrainAppConfig:
    """Project one manager config into a child run warm-started from another run."""

    train_config = build_managed_train_app_config(
        config,
        run_id=run_id,
        run_dir=run_dir,
    )
    return train_config.model_copy(
        update={
            "train": train_config.train.model_copy(
                update={
                    "resume_run_dir": source_run_dir,
                    "resume_artifact": source_artifact,
                    "resume_mode": "weights_only",
                    "tensorboard_step_offset": tensorboard_step_offset,
                }
            )
        }
    )
