# src/rl_fzerox/apps/run_manager/api/routers/drafts.py
from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, HTTPException, Path

from rl_fzerox.apps.run_manager.api import handlers
from rl_fzerox.apps.run_manager.api.contracts import CreateDraftRequest, UpdateDraftRequest
from rl_fzerox.apps.run_manager.api.execution import run_sync
from rl_fzerox.core.manager import ManagerStore


def create_drafts_router(store: ManagerStore) -> APIRouter:
    router = APIRouter()

    @router.get("/api/drafts")
    async def drafts() -> dict[str, list[dict[str, object]]]:
        return await run_sync(handlers.drafts_payload, store)

    @router.post("/api/drafts", status_code=201)
    async def create_draft(request: CreateDraftRequest) -> dict[str, dict[str, object]]:
        name = request.name.strip()
        if not name:
            raise HTTPException(status_code=400, detail="draft name is required")
        handlers.validate_source_fields(
            source_run_id=request.source_run_id,
            source_artifact=request.source_artifact,
        )
        return await run_sync(handlers.create_draft_payload, store, request, name)

    @router.put("/api/drafts/{draft_id}")
    async def update_draft(
        draft_id: Annotated[str, Path(min_length=1)],
        request: UpdateDraftRequest,
    ) -> dict[str, dict[str, object]]:
        name = request.name.strip()
        if not name:
            raise HTTPException(status_code=400, detail="draft name is required")
        handlers.validate_source_fields(
            source_run_id=request.source_run_id,
            source_artifact=request.source_artifact,
        )
        return await run_sync(handlers.update_draft_payload, store, draft_id, request, name)

    @router.delete("/api/drafts/{draft_id}")
    async def delete_draft(
        draft_id: Annotated[str, Path(min_length=1)],
    ) -> dict[str, bool]:
        return await run_sync(handlers.delete_draft_payload, store, draft_id)

    return router
