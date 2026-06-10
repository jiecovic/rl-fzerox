# src/rl_fzerox/apps/run_manager/api/routers/lineages.py
from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Path

from rl_fzerox.apps.run_manager.api import handlers
from rl_fzerox.apps.run_manager.api.contracts import UpdateLineageGroupsRequest
from rl_fzerox.apps.run_manager.api.execution import run_sync
from rl_fzerox.core.manager import ManagerStore


def create_lineages_router(store: ManagerStore) -> APIRouter:
    router = APIRouter()

    @router.delete("/api/lineages/{lineage_id}")
    async def delete_lineage(lineage_id: Annotated[str, Path(min_length=1)]) -> dict[str, bool]:
        return await run_sync(handlers.delete_lineage_payload, store, lineage_id)

    @router.put("/api/lineages/{lineage_id}/groups")
    async def update_lineage_groups(
        lineage_id: Annotated[str, Path(min_length=1)],
        request: UpdateLineageGroupsRequest,
    ) -> dict[str, object]:
        return await run_sync(handlers.update_lineage_groups_payload, store, lineage_id, request)

    return router
