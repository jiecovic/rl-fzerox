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

    @router.get("/api/runs/{run_id}/track-sampling/alt-baselines")
    async def run_alt_baselines(
        run_id: Annotated[str, Path(min_length=1)],
    ) -> dict[str, object]:
        return await run_sync(handlers.run_alt_baselines_payload, store, run_id)

    @router.get("/api/runs/{run_id}/engine-tuning")
    async def run_engine_tuning(
        run_id: Annotated[str, Path(min_length=1)],
        artifact: Literal["latest", "best"] = Query(default="latest"),
    ) -> dict[str, object]:
        return await run_sync(
            handlers.run_engine_tuning_payload,
            store,
            run_id,
            artifact=artifact,
        )

    @router.post("/api/runs/{run_id}/track-sampling/reset")
    async def reset_run_track_sampling(
        run_id: Annotated[str, Path(min_length=1)],
    ) -> dict[str, bool]:
        return await run_sync(handlers.reset_run_track_sampling_payload, store, run_id)

    @router.post("/api/runs/{run_id}/engine-tuning/reset")
    async def reset_run_engine_tuning(
        run_id: Annotated[str, Path(min_length=1)],
    ) -> dict[str, object]:
        return await run_sync(handlers.reset_run_engine_tuning_payload, store, run_id)

    @router.delete("/api/runs/{run_id}/track-sampling/alt-baselines")
    async def clear_run_alt_baselines(
        run_id: Annotated[str, Path(min_length=1)],
    ) -> dict[str, object]:
        return await run_sync(handlers.clear_run_alt_baselines_payload, store, run_id)

    @router.delete("/api/runs/{run_id}/track-sampling/alt-baselines/course")
    async def clear_run_course_alt_baselines(
        run_id: Annotated[str, Path(min_length=1)],
        course_key: Annotated[str, Query(min_length=1)],
    ) -> dict[str, object]:
        return await run_sync(
            handlers.clear_run_course_alt_baselines_payload,
            store,
            run_id,
            course_key=course_key,
        )

    @router.post("/api/tensorboard-views/rebuild")
    async def rebuild_tensorboard_views_endpoint() -> dict[str, object]:
        return await run_sync(handlers.rebuild_tensorboard_views_payload, store)

    return router
