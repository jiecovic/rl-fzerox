# src/rl_fzerox/core/manager/__init__.py
"""Public manager package surface.

External callers should usually enter through:
- ``ManagedRunConfig`` for the canonical SQLite-backed run-spec model
- ``ManagerStore`` for persistence and runtime coordination
- manager data models for drafts, runs, events, and runtime snapshots
"""

from rl_fzerox.core.manager.errors import ManagerNameConflictError
from rl_fzerox.core.manager.models import (
    ManagedRun,
    ManagedRunDraft,
    ManagedRunEvent,
    ManagedRunMetricSample,
    ManagedRunRuntime,
    ManagedRunSummary,
    ManagedRunTemplate,
)
from rl_fzerox.core.manager.run_spec import (
    ManagedRunConfig,
    default_managed_run_config,
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
    "ManagedRunSummary",
    "ManagedRunTemplate",
    "ManagerStore",
    "default_managed_run_config",
    "default_manager_db_path",
    "new_managed_run_id",
]
