# src/rl_fzerox/core/manager/registry/evaluations/__init__.py
"""Facade for manager evaluation registry operations."""

from __future__ import annotations

from rl_fzerox.core.manager.registry.evaluations.baselines import (
    baseline_suite_for_evaluation,
    list_evaluation_baseline_suites,
)
from rl_fzerox.core.manager.registry.evaluations.presets import (
    create_evaluation_preset,
    delete_evaluation_preset,
    get_evaluation_preset,
    list_evaluation_presets,
)
from rl_fzerox.core.manager.registry.evaluations.records import (
    CANCEL_REQUEST_FILENAME,
    create_evaluation,
    delete_evaluation,
    evaluation_cancel_request_path,
    get_evaluation,
    list_evaluations,
    mark_evaluation_cancelled,
    mark_evaluation_completed,
    mark_evaluation_failed,
    mark_evaluation_running,
    request_evaluation_cancel,
    update_evaluation_name,
)

__all__ = [
    "CANCEL_REQUEST_FILENAME",
    "baseline_suite_for_evaluation",
    "create_evaluation",
    "create_evaluation_preset",
    "delete_evaluation",
    "delete_evaluation_preset",
    "evaluation_cancel_request_path",
    "get_evaluation",
    "get_evaluation_preset",
    "list_evaluation_baseline_suites",
    "list_evaluation_presets",
    "list_evaluations",
    "mark_evaluation_cancelled",
    "mark_evaluation_completed",
    "mark_evaluation_failed",
    "mark_evaluation_running",
    "request_evaluation_cancel",
    "update_evaluation_name",
]
