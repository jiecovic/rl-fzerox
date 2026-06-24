# src/rl_fzerox/apps/run_manager/api/routers/evaluations.py
from __future__ import annotations

from fastapi import APIRouter

from rl_fzerox.apps.run_manager.api import handlers
from rl_fzerox.apps.run_manager.api.contracts import (
    CreateEvaluationPresetRequest,
    CreateEvaluationRequest,
    StartEvaluationRequest,
    UpdateEvaluationRequest,
)
from rl_fzerox.apps.run_manager.api.execution import run_sync
from rl_fzerox.apps.run_manager.api.payloads.evaluations import (
    DeleteResponsePayload,
    EvaluationPresetResponsePayload,
    EvaluationResponsePayload,
    EvaluationsPayload,
)
from rl_fzerox.apps.run_manager.api.validation import required_name
from rl_fzerox.core.manager import ManagerStore


def create_evaluations_router(store: ManagerStore) -> APIRouter:
    router = APIRouter()

    @router.get("/api/evaluations")
    async def evaluations() -> EvaluationsPayload:
        return await run_sync(handlers.evaluations_payload, store)

    @router.post("/api/evaluations", status_code=201)
    async def create_evaluation(
        request: CreateEvaluationRequest,
    ) -> EvaluationResponsePayload:
        name = required_name(request.name, subject="evaluation")
        return await run_sync(handlers.create_evaluation_payload, store, request, name)

    @router.post("/api/evaluation-presets", status_code=201)
    async def create_evaluation_preset(
        request: CreateEvaluationPresetRequest,
    ) -> EvaluationPresetResponsePayload:
        name = required_name(request.name, subject="evaluation preset")
        return await run_sync(handlers.create_evaluation_preset_payload, store, request, name)

    @router.delete("/api/evaluation-presets/{preset_id}")
    async def delete_evaluation_preset(preset_id: str) -> DeleteResponsePayload:
        return await run_sync(handlers.delete_evaluation_preset_payload, store, preset_id)

    @router.post("/api/evaluations/{evaluation_id}/start")
    async def start_evaluation(
        evaluation_id: str,
        request: StartEvaluationRequest | None = None,
    ) -> EvaluationResponsePayload:
        return await run_sync(
            handlers.start_evaluation_payload,
            store,
            evaluation_id,
            request or StartEvaluationRequest(),
        )

    @router.post("/api/evaluations/{evaluation_id}/cancel")
    async def cancel_evaluation(evaluation_id: str) -> EvaluationResponsePayload:
        return await run_sync(handlers.cancel_evaluation_payload, store, evaluation_id)

    @router.put("/api/evaluations/{evaluation_id}")
    async def update_evaluation(
        evaluation_id: str,
        request: UpdateEvaluationRequest,
    ) -> EvaluationResponsePayload:
        name = required_name(request.name, subject="evaluation")
        return await run_sync(handlers.update_evaluation_payload, store, evaluation_id, name)

    @router.delete("/api/evaluations/{evaluation_id}")
    async def delete_evaluation(evaluation_id: str) -> DeleteResponsePayload:
        return await run_sync(handlers.delete_evaluation_payload, store, evaluation_id)

    return router
