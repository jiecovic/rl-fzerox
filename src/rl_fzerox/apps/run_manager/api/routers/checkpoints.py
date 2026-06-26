# src/rl_fzerox/apps/run_manager/api/routers/checkpoints.py
from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Path

from rl_fzerox.apps.run_manager.api import handlers
from rl_fzerox.apps.run_manager.api.execution import run_sync
from rl_fzerox.apps.run_manager.api.payloads.checkpoints import CheckpointCatalogPayload
from rl_fzerox.core.manager import ManagerStore


def create_checkpoints_router(store: ManagerStore) -> APIRouter:
    router = APIRouter()

    @router.get("/api/checkpoints/catalog")
    async def checkpoint_catalog() -> CheckpointCatalogPayload:
        return await run_sync(handlers.checkpoint_catalog_response, store)

    @router.post("/api/checkpoints/catalog/{checkpoint_id}/{version}/install")
    async def install_catalog_checkpoint(
        checkpoint_id: Annotated[str, Path(min_length=1)],
        version: Annotated[str, Path(min_length=1)],
    ) -> dict[str, object]:
        return await run_sync(
            handlers.install_catalog_checkpoint_response,
            store,
            checkpoint_id=checkpoint_id,
            version=version,
        )

    return router
