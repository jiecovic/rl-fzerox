# src/rl_fzerox/core/career_mode/execution/reports.py
"""Serializable reports for save-game unlock execution plans.

Reports are lightweight diagnostics for manager/UI surfaces. They mirror the
resolved execution context and should not become a second source of truth.
"""

from __future__ import annotations

from dataclasses import asdict

from rl_fzerox.core.career_mode.execution.context import SaveAttemptExecutionContext
from rl_fzerox.core.career_mode.execution.race import SaveRaceExecutionPlan


def save_race_execution_plan_report(
    *,
    context: SaveAttemptExecutionContext,
    plan: SaveRaceExecutionPlan,
) -> dict[str, object]:
    """Return a stable JSON-ready summary for one resolved save-game race plan."""

    return {
        "attempt": {
            "id": context.attempt.id,
            "status": context.attempt.status,
            "target_kind": context.attempt.target_kind,
        },
        "policy": {
            "artifact": plan.policy_artifact,
            "algorithm": plan.policy_algorithm,
            "path": str(plan.policy_path),
            "source_dir": str(plan.policy_source_dir),
            "source_id": plan.policy_source_id,
            "source_kind": plan.policy_source_kind,
            "source_name": plan.policy_source_name,
        },
        "race_setup": asdict(plan.race_setup),
        "save_game": {
            "id": context.save_game.id,
            "name": context.save_game.name,
            "path": str(context.save_game.save_path),
            "status": context.save_game.status,
        },
        "target": {
            "kind": context.target.kind,
            "label": context.target.label,
            "sequence_index": context.target.sequence_index,
        },
    }
