# src/rl_fzerox/apps/run_manager/api/routers/system.py
from __future__ import annotations

from fastapi import APIRouter

from rl_fzerox.apps.run_manager.api import handlers
from rl_fzerox.apps.run_manager.api.execution import run_sync
from rl_fzerox.core.manager import ManagedRunConfig, ManagerStore


def create_system_router(store: ManagerStore) -> APIRouter:
    router = APIRouter()

    @router.get("/api/health")
    async def health() -> dict[str, bool]:
        return {"ok": True}

    @router.get("/api/templates")
    async def templates() -> dict[str, list[dict[str, object]]]:
        return await run_sync(handlers.templates_payload, store)

    @router.get("/api/config-metadata")
    async def config_metadata() -> dict[str, object]:
        return await run_sync(handlers.config_metadata_payload)

    @router.post("/api/policy-preview")
    async def policy_preview(config: ManagedRunConfig) -> dict[str, object]:
        return await run_sync(handlers.policy_preview_payload, config)

    return router
