# src/rl_fzerox/core/manager/training.py
"""Managed-training projection facade."""

from rl_fzerox.core.manager.projection.compat import assert_managed_fork_compatible
from rl_fzerox.core.manager.projection.launches import (
    build_managed_fork_train_app_config,
    build_managed_resume_train_app_config,
    build_managed_train_app_config,
)

__all__ = [
    "assert_managed_fork_compatible",
    "build_managed_fork_train_app_config",
    "build_managed_resume_train_app_config",
    "build_managed_train_app_config",
]
