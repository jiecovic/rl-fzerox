# src/rl_fzerox/core/manager/training.py
"""Managed-training projection facade."""

from rl_fzerox.core.manager.projection.compat import assert_managed_fork_compatible
from rl_fzerox.core.manager.projection.launches import (
    apply_managed_resume_train_config,
    build_managed_fork_train_app_config,
    build_managed_fork_train_app_config_from_metadata,
    build_managed_resume_train_app_config,
    build_managed_train_app_config,
)

__all__ = [
    "apply_managed_resume_train_config",
    "assert_managed_fork_compatible",
    "build_managed_fork_train_app_config",
    "build_managed_fork_train_app_config_from_metadata",
    "build_managed_resume_train_app_config",
    "build_managed_train_app_config",
]
