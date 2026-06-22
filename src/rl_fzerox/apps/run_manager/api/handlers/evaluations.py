# src/rl_fzerox/apps/run_manager/api/handlers/evaluations.py
from __future__ import annotations

from fastapi import HTTPException

from rl_fzerox.apps.run_manager.api.contracts import CreateEvaluationRequest
from rl_fzerox.apps.run_manager.api.payloads.evaluations import evaluation_payload
from rl_fzerox.core.evaluation.models import EvaluationTargetSpec
from rl_fzerox.core.manager import ManagerStore


def evaluations_payload(store: ManagerStore) -> dict[str, list[dict[str, object]]]:
    """Return all manager-owned evaluations."""

    return {
        "evaluations": [evaluation_payload(evaluation) for evaluation in store.list_evaluations()]
    }


def create_evaluation_payload(
    store: ManagerStore,
    request: CreateEvaluationRequest,
    name: str,
) -> dict[str, dict[str, object]]:
    """Create one immutable checkpoint snapshot for future evaluation execution."""

    try:
        evaluation = store.create_evaluation(
            name=name,
            source_run_id=request.source_run_id,
            source_artifact=request.source_artifact,
            policy_mode=request.policy_mode,
            seed=request.seed,
            target=EvaluationTargetSpec(
                mode=request.target.mode,
                course_ids=tuple(request.target.course_ids),
                cup_ids=tuple(request.target.cup_ids),
                difficulties=tuple(request.target.difficulties),
                vehicle_ids=tuple(request.target.vehicle_ids),
                repeats_per_target=request.target.repeats_per_target,
            ),
        )
    except FileNotFoundError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return {"evaluation": evaluation_payload(evaluation)}
