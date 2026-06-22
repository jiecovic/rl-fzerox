# src/rl_fzerox/apps/run_manager/api/routers/evaluations.py
from __future__ import annotations

from fastapi import APIRouter

from rl_fzerox.apps.run_manager.api import handlers
from rl_fzerox.apps.run_manager.api.contracts import CreateEvaluationRequest
from rl_fzerox.apps.run_manager.api.execution import run_sync
from rl_fzerox.apps.run_manager.api.validation import required_name
from rl_fzerox.core.manager import ManagerStore


def create_evaluations_router(store: ManagerStore) -> APIRouter:
    router = APIRouter()

    @router.get("/api/evaluations")
    async def evaluations() -> dict[str, list[dict[str, object]]]:
        return await run_sync(handlers.evaluations_payload, store)

    @router.post("/api/evaluations", status_code=201)
    async def create_evaluation(
        request: CreateEvaluationRequest,
    ) -> dict[str, dict[str, object]]:
        name = required_name(request.name, subject="evaluation")
        return await run_sync(handlers.create_evaluation_payload, store, request, name)

    return router
