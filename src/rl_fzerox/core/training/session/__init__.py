# src/rl_fzerox/core/training/session/__init__.py
"""Lazy training-session facade."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rl_fzerox.core.training.session.artifacts import (
        PolicyArtifactMetadata,
        cleanup_failed_run,
        current_policy_artifact_metadata,
        engine_tuning_checkpoint_path,
        load_engine_tuning_checkpoint_state,
        load_policy_artifact_metadata,
        resolve_train_run_config,
        save_artifacts_atomically,
        save_latest_artifacts,
        validate_training_baseline_state,
    )
    from rl_fzerox.core.training.session.callbacks import build_callbacks
    from rl_fzerox.core.training.session.env import build_training_env
    from rl_fzerox.core.training.session.model import (
        build_ppo_model,
        build_tensorboard_logger,
        build_training_model,
        maybe_resume_training_model,
        print_training_startup,
        training_requires_action_masks,
        validate_training_algorithm_config,
    )


_EXPORT_MODULES = {
    "PolicyArtifactMetadata": "rl_fzerox.core.training.session.artifacts",
    "build_callbacks": "rl_fzerox.core.training.session.callbacks",
    "build_ppo_model": "rl_fzerox.core.training.session.model",
    "build_tensorboard_logger": "rl_fzerox.core.training.session.model",
    "build_training_env": "rl_fzerox.core.training.session.env",
    "build_training_model": "rl_fzerox.core.training.session.model",
    "cleanup_failed_run": "rl_fzerox.core.training.session.artifacts",
    "current_policy_artifact_metadata": "rl_fzerox.core.training.session.artifacts",
    "engine_tuning_checkpoint_path": "rl_fzerox.core.training.session.artifacts",
    "load_engine_tuning_checkpoint_state": "rl_fzerox.core.training.session.artifacts",
    "load_policy_artifact_metadata": "rl_fzerox.core.training.session.artifacts",
    "maybe_resume_training_model": "rl_fzerox.core.training.session.model",
    "print_training_startup": "rl_fzerox.core.training.session.model",
    "resolve_train_run_config": "rl_fzerox.core.training.session.artifacts",
    "save_artifacts_atomically": "rl_fzerox.core.training.session.artifacts",
    "save_latest_artifacts": "rl_fzerox.core.training.session.artifacts",
    "training_requires_action_masks": "rl_fzerox.core.training.session.model",
    "validate_training_algorithm_config": "rl_fzerox.core.training.session.model",
    "validate_training_baseline_state": "rl_fzerox.core.training.session.artifacts",
}

__all__ = [
    "build_callbacks",
    "build_ppo_model",
    "build_tensorboard_logger",
    "build_training_env",
    "build_training_model",
    "cleanup_failed_run",
    "current_policy_artifact_metadata",
    "engine_tuning_checkpoint_path",
    "load_engine_tuning_checkpoint_state",
    "maybe_resume_training_model",
    "print_training_startup",
    "load_policy_artifact_metadata",
    "PolicyArtifactMetadata",
    "resolve_train_run_config",
    "save_artifacts_atomically",
    "save_latest_artifacts",
    "training_requires_action_masks",
    "validate_training_algorithm_config",
    "validate_training_baseline_state",
]


def __getattr__(name: str) -> object:
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(__all__)
