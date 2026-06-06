# src/rl_fzerox/core/manager/__init__.py
"""Lazy public manager package surface.

External callers should usually enter through:
- ``ManagedRunConfig`` for the canonical SQLite-backed run-spec model
- ``ManagerStore`` for persistence and runtime coordination
- manager data models for drafts, runs, events, and runtime snapshots
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rl_fzerox.core.manager.errors import ManagerNameConflictError
    from rl_fzerox.core.manager.models import (
        CourseSetupScope,
        ManagedRun,
        ManagedRunDraft,
        ManagedRunEvent,
        ManagedRunMetricSample,
        ManagedRunRuntime,
        ManagedRunSummary,
        ManagedRunTemplate,
        ManagedSaveAttempt,
        ManagedSaveCourseSetup,
        ManagedSaveGame,
        ManagedSaveUnlockProgress,
        ManagedSaveUnlockTarget,
        ManagedViewerLease,
        ViewerLeaseKind,
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

_EXPORT_MODULES = {
    "ManagerNameConflictError": "rl_fzerox.core.manager.errors",
    "ManagedRun": "rl_fzerox.core.manager.models",
    "ManagedRunConfig": "rl_fzerox.core.manager.run_spec",
    "ManagedRunDraft": "rl_fzerox.core.manager.models",
    "ManagedRunEvent": "rl_fzerox.core.manager.models",
    "ManagedRunMetricSample": "rl_fzerox.core.manager.models",
    "ManagedRunRuntime": "rl_fzerox.core.manager.models",
    "ManagedRunSummary": "rl_fzerox.core.manager.models",
    "ManagedRunTemplate": "rl_fzerox.core.manager.models",
    "ManagedSaveAttempt": "rl_fzerox.core.manager.models",
    "ManagedSaveGame": "rl_fzerox.core.manager.models",
    "ManagedSaveCourseSetup": "rl_fzerox.core.manager.models",
    "ManagedSaveUnlockProgress": "rl_fzerox.core.manager.models",
    "ManagedSaveUnlockTarget": "rl_fzerox.core.manager.models",
    "ManagedViewerLease": "rl_fzerox.core.manager.models",
    "ManagerStore": "rl_fzerox.core.manager.store",
    "CourseSetupScope": "rl_fzerox.core.manager.models",
    "ViewerLeaseKind": "rl_fzerox.core.manager.models",
    "default_managed_run_config": "rl_fzerox.core.manager.run_spec",
    "default_manager_db_path": "rl_fzerox.core.manager.store",
    "new_managed_run_id": "rl_fzerox.core.manager.store",
}

__all__ = [
    "ManagerNameConflictError",
    "ManagedRun",
    "ManagedRunConfig",
    "ManagedRunDraft",
    "ManagedRunEvent",
    "ManagedRunMetricSample",
    "ManagedRunRuntime",
    "ManagedRunSummary",
    "ManagedRunTemplate",
    "ManagedSaveAttempt",
    "ManagedSaveGame",
    "ManagedSaveCourseSetup",
    "ManagedSaveUnlockProgress",
    "ManagedSaveUnlockTarget",
    "ManagedViewerLease",
    "ManagerStore",
    "CourseSetupScope",
    "ViewerLeaseKind",
    "default_managed_run_config",
    "default_manager_db_path",
    "new_managed_run_id",
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
