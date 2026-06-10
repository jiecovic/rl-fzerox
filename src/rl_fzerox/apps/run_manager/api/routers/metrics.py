# src/rl_fzerox/apps/run_manager/api/routers/metrics.py
from __future__ import annotations

from typing import Annotated, Literal

from fastapi import APIRouter, Path, Query

from rl_fzerox.apps.run_manager.api import handlers
from rl_fzerox.apps.run_manager.api.execution import run_sync
from rl_fzerox.core.manager import ManagerStore


def create_metrics_router(store: ManagerStore) -> APIRouter:
    router = APIRouter()

    @router.get("/api/runs/{run_id}/metrics")
    async def run_metrics(
        run_id: Annotated[str, Path(min_length=1)],
        mode: Literal["recent", "full"] = Query(default="recent"),
        limit: int = Query(default=240, ge=1, le=2_000),
    ) -> dict[str, list[dict[str, object]]]:
        return await run_sync(
            handlers.run_metrics_payload,
            store,
            run_id,
            limit=None if mode == "full" else limit,
        )

    @router.get("/api/runs/{run_id}/track-sampling")
    async def run_track_sampling(
        run_id: Annotated[str, Path(min_length=1)],
    ) -> dict[str, object]:
        return await run_sync(handlers.run_track_sampling_payload, store, run_id)

    @router.post("/api/runs/{run_id}/track-sampling/reset")
    async def reset_run_track_sampling(
        run_id: Annotated[str, Path(min_length=1)],
    ) -> dict[str, bool]:
        return await run_sync(handlers.reset_run_track_sampling_payload, store, run_id)

    @router.post("/api/tensorboard-views/rebuild")
    async def rebuild_tensorboard_views_endpoint() -> dict[str, object]:
        return await run_sync(handlers.rebuild_tensorboard_views_payload, store)

    return router
