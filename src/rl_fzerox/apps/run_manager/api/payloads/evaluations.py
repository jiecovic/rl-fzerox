# src/rl_fzerox/apps/run_manager/api/payloads/evaluations.py
from __future__ import annotations

import json
from dataclasses import asdict

from rl_fzerox.core.evaluation.models import EvaluationCheckpointSnapshot
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
        "config": evaluation.config.model_dump(mode="json"),
        "checkpoint": _checkpoint_payload(evaluation.checkpoint),
        "created_at": evaluation.created_at,
        "updated_at": evaluation.updated_at,
        "started_at": evaluation.started_at,
        "finished_at": evaluation.finished_at,
        "result_json_path": (
            None if evaluation.result_json_path is None else str(evaluation.result_json_path)
        ),
        "error_message": evaluation.error_message,
        "progress": _progress_payload(evaluation),
    }


def _checkpoint_payload(checkpoint: EvaluationCheckpointSnapshot) -> dict[str, object]:
    payload = asdict(checkpoint)
    # Nanosecond mtimes exceed JavaScript's safe integer range. Keep the Python
    # model and persisted spec numeric, but expose the API value losslessly.
    payload["source_mtime_ns"] = (
        None if checkpoint.source_mtime_ns is None else str(checkpoint.source_mtime_ns)
    )
    return payload


def _progress_payload(evaluation: ManagedEvaluation) -> dict[str, object]:
    completed_attempts = 0
    total_attempts: int | None = None
    result_status: str | None = None
    if evaluation.result_json_path is not None and evaluation.result_json_path.is_file():
        try:
            payload = json.loads(evaluation.result_json_path.read_text(encoding="utf-8"))
            result = payload.get("result") if isinstance(payload, dict) else None
            if isinstance(result, dict):
                attempts = result.get("attempts")
                if isinstance(attempts, list):
                    completed_attempts = len(attempts)
                spec = result.get("spec")
                if isinstance(spec, dict):
                    planned = spec.get("total_planned_attempts")
                    if isinstance(planned, int) and planned >= 0:
                        total_attempts = planned
                raw_status = result.get("status")
                if isinstance(raw_status, str):
                    result_status = raw_status
        except (OSError, json.JSONDecodeError):
            completed_attempts = 0
    return {
        "completed_attempts": completed_attempts,
        "total_attempts": total_attempts,
        "result_status": result_status,
    }
