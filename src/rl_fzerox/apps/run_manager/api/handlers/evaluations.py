# src/rl_fzerox/apps/run_manager/api/handlers/evaluations.py
from __future__ import annotations

from fastapi import HTTPException

from rl_fzerox.apps.run_manager.api.contracts import (
    CreateEvaluationPresetRequest,
    CreateEvaluationRequest,
    StartEvaluationRequest,
)
from rl_fzerox.apps.run_manager.api.payloads.evaluations import (
    DeleteResponsePayload,
    EvaluationPayload,
    EvaluationPresetResponsePayload,
    EvaluationResponsePayload,
    EvaluationsPayload,
    evaluation_baseline_suite_payload,
    evaluation_payload,
    evaluation_preset_payload,
)
from rl_fzerox.apps.run_manager.launching import launch_evaluation_worker
from rl_fzerox.core.evaluation.models import EvaluationTargetSpec
from rl_fzerox.core.manager import ManagedEvaluation, ManagedEvaluationBaselineSuite, ManagerStore


def evaluations_payload(store: ManagerStore) -> EvaluationsPayload:
    """Return all manager-owned evaluations."""

    baseline_suites = store.list_evaluation_baseline_suites()
    suites_by_preset = {(suite.preset_id, suite.preset_version): suite for suite in baseline_suites}
    return {
        "evaluations": [
            evaluation_payload(
                evaluation,
                baseline_suite=_required_suite_for_evaluation(evaluation, suites_by_preset),
            )
            for evaluation in store.list_evaluations()
        ],
        "presets": [
            evaluation_preset_payload(preset) for preset in store.list_evaluation_presets()
        ],
        "baseline_suites": [evaluation_baseline_suite_payload(suite) for suite in baseline_suites],
    }


def create_evaluation_payload(
    store: ManagerStore,
    request: CreateEvaluationRequest,
    name: str,
) -> EvaluationResponsePayload:
    """Create one immutable checkpoint snapshot for future evaluation execution."""

    try:
        evaluation = store.create_evaluation_from_policy_source(
            name=name,
            source_policy_kind=request.source_policy_kind,
            source_policy_id=_evaluation_source_policy_id(request),
            source_artifact=request.source_artifact,
            policy_mode=request.policy_mode,
            preset_id=request.preset_id,
        )
    except FileNotFoundError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return {"evaluation": _evaluation_payload_with_suite(store, evaluation)}


def _evaluation_source_policy_id(request: CreateEvaluationRequest) -> str:
    if request.source_policy_id is not None:
        return request.source_policy_id
    if request.source_policy_kind == "run" and request.source_run_id is not None:
        return request.source_run_id
    raise ValueError("source_policy_id is required")


def create_evaluation_preset_payload(
    store: ManagerStore,
    request: CreateEvaluationPresetRequest,
    name: str,
) -> EvaluationPresetResponsePayload:
    """Create one custom immutable benchmark preset."""

    target = EvaluationTargetSpec(
        mode=request.target.mode,
        course_ids=tuple(request.target.course_ids),
        cup_ids=tuple(request.target.cup_ids),
        difficulties=tuple(request.target.difficulties),
        repeats_per_target=request.target.repeats_per_target,
        baseline_variant_count=request.target.baseline_variant_count,
    )
    try:
        preset = store.create_evaluation_preset(
            name=name,
            seed=request.seed,
            renderer=request.renderer,
            target=target,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return {"preset": evaluation_preset_payload(preset)}


def delete_evaluation_preset_payload(
    store: ManagerStore,
    preset_id: str,
) -> DeleteResponsePayload:
    """Delete one unused custom benchmark preset."""

    try:
        deleted = store.delete_evaluation_preset(preset_id)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return {"deleted": deleted}


def delete_evaluation_payload(store: ManagerStore, evaluation_id: str) -> DeleteResponsePayload:
    """Delete one created evaluation snapshot."""

    try:
        deleted = store.delete_evaluation(evaluation_id)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return {"deleted": deleted}


def update_evaluation_payload(
    store: ManagerStore,
    evaluation_id: str,
    name: str,
) -> EvaluationResponsePayload:
    """Rename one manager-owned evaluation."""

    try:
        evaluation = store.update_evaluation_name(evaluation_id=evaluation_id, name=name)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    if evaluation is None:
        raise HTTPException(status_code=404, detail="evaluation not found")
    return {"evaluation": _evaluation_payload_with_suite(store, evaluation)}


def start_evaluation_payload(
    store: ManagerStore,
    evaluation_id: str,
    request: StartEvaluationRequest,
) -> EvaluationResponsePayload:
    """Start or retry one inactive evaluation snapshot."""

    try:
        evaluation = launch_evaluation_worker(
            store,
            evaluation_id=evaluation_id,
            device=request.device,
            worker_count=request.worker_count,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return {"evaluation": _evaluation_payload_with_suite(store, evaluation)}


def cancel_evaluation_payload(
    store: ManagerStore,
    evaluation_id: str,
) -> EvaluationResponsePayload:
    """Request cooperative cancellation for one running evaluation."""

    try:
        evaluation = store.request_evaluation_cancel(evaluation_id)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    if evaluation is None:
        raise HTTPException(status_code=404, detail="evaluation not found")
    return {"evaluation": _evaluation_payload_with_suite(store, evaluation)}


def _evaluation_payload_with_suite(
    store: ManagerStore,
    evaluation: ManagedEvaluation,
) -> EvaluationPayload:
    suites_by_preset = {
        (suite.preset_id, suite.preset_version): suite
        for suite in store.list_evaluation_baseline_suites()
    }
    return evaluation_payload(
        evaluation,
        baseline_suite=_required_suite_for_evaluation(evaluation, suites_by_preset),
    )


def _required_suite_for_evaluation(
    evaluation: ManagedEvaluation,
    suites_by_preset: dict[tuple[str, int], ManagedEvaluationBaselineSuite],
) -> ManagedEvaluationBaselineSuite:
    suite = suites_by_preset.get((evaluation.preset_id, evaluation.preset_version))
    if suite is None:
        raise HTTPException(status_code=500, detail="evaluation baseline suite is missing")
    return suite
