# src/rl_fzerox/core/manager/db/repositories/evaluations/__init__.py
"""Facade for evaluation repository operations."""

from __future__ import annotations

from rl_fzerox.core.manager.db.repositories.evaluations.baselines import (
    ensure_evaluation_baseline_suite,
    list_evaluation_baseline_suites,
    upsert_evaluation_baseline_suite_status,
)
from rl_fzerox.core.manager.db.repositories.evaluations.mapping import (
    evaluation_baseline_suite_from_model,
    evaluation_preset_from_model,
    managed_evaluation_from_model,
)
from rl_fzerox.core.manager.db.repositories.evaluations.presets import (
    delete_evaluation_preset,
    get_evaluation_preset,
    insert_evaluation_preset,
    list_evaluation_presets,
    upsert_default_evaluation_presets,
)
from rl_fzerox.core.manager.db.repositories.evaluations.records import (
    delete_inactive_evaluation,
    find_created_evaluation_snapshot,
    get_managed_evaluation,
    insert_evaluation,
    list_managed_evaluations,
    mark_evaluation_cancelled,
    mark_evaluation_cancelling,
    mark_evaluation_completed,
    mark_evaluation_failed,
    mark_evaluation_running,
    update_evaluation_name,
)

__all__ = [
    "delete_evaluation_preset",
    "delete_inactive_evaluation",
    "ensure_evaluation_baseline_suite",
    "evaluation_baseline_suite_from_model",
    "evaluation_preset_from_model",
    "find_created_evaluation_snapshot",
    "get_evaluation_preset",
    "get_managed_evaluation",
    "insert_evaluation",
    "insert_evaluation_preset",
    "list_evaluation_baseline_suites",
    "list_evaluation_presets",
    "list_managed_evaluations",
    "managed_evaluation_from_model",
    "mark_evaluation_cancelled",
    "mark_evaluation_cancelling",
    "mark_evaluation_completed",
    "mark_evaluation_failed",
    "mark_evaluation_running",
    "update_evaluation_name",
    "upsert_default_evaluation_presets",
    "upsert_evaluation_baseline_suite_status",
]
