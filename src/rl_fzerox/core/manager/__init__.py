# src/rl_fzerox/core/manager/__init__.py
"""SQLite-backed run manager primitives."""

from rl_fzerox.core.manager.config import ManagedRunConfig, default_managed_run_config
from rl_fzerox.core.manager.errors import ManagerNameConflictError
from rl_fzerox.core.manager.models import (
    ManagedRun,
    ManagedRunDraft,
    ManagedRunEvent,
    ManagedRunMetricSample,
    ManagedRunRuntime,
    ManagedRunTemplate,
)
from rl_fzerox.core.manager.store import (
    ManagerStore,
    default_manager_db_path,
    new_managed_run_id,
)

__all__ = [
    "ManagerNameConflictError",
    "ManagedRunConfig",
    "ManagedRun",
    "ManagedRunDraft",
    "ManagedRunEvent",
    "ManagedRunMetricSample",
    "ManagedRunRuntime",
    "ManagedRunTemplate",
    "ManagerStore",
    "default_managed_run_config",
    "default_manager_db_path",
    "new_managed_run_id",
]
