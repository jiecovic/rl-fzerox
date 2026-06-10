# src/rl_fzerox/apps/run_manager/api/routers/streams.py
from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Path, WebSocket

from rl_fzerox.apps.run_manager.api.live import (
    KeyedLiveSnapshotBroadcaster,
    LiveSnapshotBroadcaster,
)


def create_streams_router(
    *,
    live_broadcaster: LiveSnapshotBroadcaster,
    track_sampling_live_broadcaster: KeyedLiveSnapshotBroadcaster,
) -> APIRouter:
    router = APIRouter()

    @router.websocket("/api/runs/live")
    async def live_runs(websocket: WebSocket) -> None:
        await live_broadcaster.serve(websocket)

    @router.websocket("/api/runs/{run_id}/track-sampling/live")
    async def live_run_track_sampling(
        websocket: WebSocket,
        run_id: Annotated[str, Path(min_length=1)],
    ) -> None:
        await track_sampling_live_broadcaster.serve(run_id, websocket)

    return router
