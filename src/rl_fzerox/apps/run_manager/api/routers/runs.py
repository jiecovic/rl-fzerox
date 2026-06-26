# src/rl_fzerox/apps/run_manager/api/routers/runs.py
from __future__ import annotations

from typing import Annotated, Literal

from fastapi import APIRouter, Path, Query

from rl_fzerox.apps.run_manager.api import handlers
from rl_fzerox.apps.run_manager.api.contracts import (
    ForkRunRequest,
    LaunchRunRequest,
    RunLauncher,
    UpdateRunRequest,
    WatchRenderer,
    WatchRunRequest,
)
from rl_fzerox.apps.run_manager.api.execution import run_sync
from rl_fzerox.apps.run_manager.api.validation import required_name
from rl_fzerox.core.manager import ManagerStore


def create_runs_router(store: ManagerStore, launcher: RunLauncher) -> APIRouter:
    router = APIRouter()

    @router.get("/api/runs")
    async def runs() -> dict[str, list[dict[str, object]]]:
        return await run_sync(handlers.runs_payload, store)

    @router.get("/api/runs/{run_id}")
    async def run_detail(
        run_id: Annotated[str, Path(min_length=1)],
    ) -> dict[str, dict[str, object]]:
        return await run_sync(handlers.run_response_for_id, store, run_id)

    @router.put("/api/runs/{run_id}")
    async def update_run(
        run_id: Annotated[str, Path(min_length=1)],
        request: UpdateRunRequest,
    ) -> dict[str, dict[str, object]]:
        name = required_name(request.name, subject="run")
        return await run_sync(handlers.update_run_payload, store, run_id, name)

    @router.post("/api/runs", status_code=201)
    async def launch_run(request: LaunchRunRequest) -> dict[str, dict[str, object]]:
        name = required_name(request.name, subject="run")
        return await run_sync(handlers.launch_run_payload, store, launcher, request, name)

    @router.post("/api/runs/{run_id}/fork", status_code=201)
    async def fork_run(
        run_id: Annotated[str, Path(min_length=1)],
        request: ForkRunRequest,
    ) -> dict[str, dict[str, object]]:
        return await run_sync(handlers.fork_run_payload, store, launcher, run_id, request)

    @router.post("/api/runs/{run_id}/pause")
    async def pause_run(
        run_id: Annotated[str, Path(min_length=1)],
    ) -> dict[str, dict[str, object]]:
        return await run_sync(handlers.pause_run_payload, store, launcher, run_id)

    @router.post("/api/runs/{run_id}/stop")
    async def stop_run(
        run_id: Annotated[str, Path(min_length=1)],
    ) -> dict[str, dict[str, object]]:
        return await run_sync(handlers.stop_run_payload, store, launcher, run_id)

    @router.post("/api/runs/{run_id}/resume")
    async def resume_run(
        run_id: Annotated[str, Path(min_length=1)],
    ) -> dict[str, dict[str, object]]:
        return await run_sync(handlers.resume_run_payload, store, launcher, run_id)

    @router.delete("/api/runs/{run_id}")
    async def delete_run(run_id: Annotated[str, Path(min_length=1)]) -> dict[str, bool]:
        return await run_sync(handlers.delete_run_payload, store, run_id)

    @router.post("/api/runs/{run_id}/open-dir")
    async def open_run_dir(run_id: Annotated[str, Path(min_length=1)]) -> dict[str, bool]:
        return await run_sync(handlers.open_run_dir_payload, store, run_id)

    @router.post("/api/runs/{run_id}/watch")
    async def watch_run(
        run_id: Annotated[str, Path(min_length=1)],
        request: WatchRunRequest | None = None,
        artifact: str = Query(default="latest"),
    ) -> dict[str, str]:
        device: Literal["cpu", "cuda"] = "cuda" if request is None else request.device
        renderer: WatchRenderer | None = None if request is None else request.renderer
        deterministic_policy = True if request is None else request.policy_mode == "deterministic"
        return await run_sync(
            handlers.watch_run_payload,
            launcher,
            run_id,
            artifact,
            device,
            renderer,
            deterministic_policy,
        )

    return router
