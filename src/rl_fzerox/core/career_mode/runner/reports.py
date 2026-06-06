# src/rl_fzerox/core/career_mode/runner/reports.py
"""Serializable reports for save-game unlock execution plans."""

from __future__ import annotations

from dataclasses import asdict

from rl_fzerox.core.career_mode.runner.context import SaveAttemptExecutionContext
from rl_fzerox.core.career_mode.runner.race import SaveRaceExecutionPlan


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
            "run_dir": str(plan.policy_run_dir),
            "run_id": plan.policy_run_id,
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
