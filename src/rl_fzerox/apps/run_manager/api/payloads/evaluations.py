# src/rl_fzerox/apps/run_manager/api/payloads/evaluations.py
from __future__ import annotations

from dataclasses import asdict

from rl_fzerox.core.manager import ManagedEvaluation


def evaluation_payload(evaluation: ManagedEvaluation) -> dict[str, object]:
    """Return one evaluation record payload."""

    return {
        "id": evaluation.id,
        "name": evaluation.name,
        "status": evaluation.status,
        "evaluation_dir": str(evaluation.evaluation_dir),
        "source_run_id": evaluation.source_run_id,
        "source_artifact": evaluation.source_artifact,
        "policy_mode": evaluation.policy_mode,
        "seed": evaluation.seed,
        "target": asdict(evaluation.target),
        "checkpoint": asdict(evaluation.checkpoint),
        "created_at": evaluation.created_at,
        "updated_at": evaluation.updated_at,
        "started_at": evaluation.started_at,
        "finished_at": evaluation.finished_at,
        "result_json_path": (
            None if evaluation.result_json_path is None else str(evaluation.result_json_path)
        ),
        "error_message": evaluation.error_message,
    }
