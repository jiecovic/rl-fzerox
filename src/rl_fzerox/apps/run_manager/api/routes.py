# src/rl_fzerox/apps/run_manager/api/routes.py
from __future__ import annotations

import asyncio
from collections.abc import Callable
from functools import partial
from typing import Annotated, Literal, ParamSpec, TypeVar

from fastapi import FastAPI, File, HTTPException, Path, Query, UploadFile, WebSocket
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from starlette.requests import Request

from rl_fzerox.apps.run_manager.api import handlers
from rl_fzerox.apps.run_manager.api.contracts import (
    CreateDraftRequest,
    ForkRunRequest,
    LaunchRunRequest,
    RunLauncher,
    UpdateDraftRequest,
    UpdateLineageGroupsRequest,
    UpdateRunRequest,
    WatchRenderer,
    WatchRunRequest,
)
from rl_fzerox.apps.run_manager.api.live import RunLiveBroadcaster
from rl_fzerox.apps.run_manager.launch import ManagerRunLauncher
from rl_fzerox.core.manager import ManagedRunConfig, ManagerStore

_P = ParamSpec("_P")
_T = TypeVar("_T")


def create_manager_api_app(
    store: ManagerStore,
    *,
    run_launcher: RunLauncher | None = None,
) -> FastAPI:
    """Create the local REST API app for the run manager."""

    store.initialize()
    store.rebuild_tensorboard_views()
    app = FastAPI(title="F-Zero X Run Manager", version="0.1.0")
    launcher = run_launcher or ManagerRunLauncher(store)
    live_broadcaster = RunLiveBroadcaster(lambda: _run_sync(handlers.runs_payload, store))

    @app.exception_handler(HTTPException)
    async def handle_http_exception(_request: Request, exc: HTTPException) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})

    @app.exception_handler(RequestValidationError)
    async def handle_validation_exception(
        _request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        return JSONResponse(status_code=400, content={"error": jsonable_encoder(exc.errors())})

    @app.get("/api/health")
    async def health() -> dict[str, bool]:
        return {"ok": True}

    @app.get("/api/templates")
    async def templates() -> dict[str, list[dict[str, object]]]:
        return await _run_sync(handlers.templates_payload, store)

    @app.get("/api/drafts")
    async def drafts() -> dict[str, list[dict[str, object]]]:
        return await _run_sync(handlers.drafts_payload, store)

    @app.get("/api/runs")
    async def runs() -> dict[str, list[dict[str, object]]]:
        return await _run_sync(handlers.runs_payload, store)

    @app.websocket("/api/runs/live")
    async def live_runs(websocket: WebSocket) -> None:
        await live_broadcaster.serve(websocket)

    @app.get("/api/runs/{run_id}")
    async def run_detail(
        run_id: Annotated[str, Path(min_length=1)],
    ) -> dict[str, dict[str, object]]:
        return await _run_sync(handlers.run_response_for_id, store, run_id)

    @app.put("/api/runs/{run_id}")
    async def update_run(
        run_id: Annotated[str, Path(min_length=1)],
        request: UpdateRunRequest,
    ) -> dict[str, dict[str, object]]:
        name = request.name.strip()
        if not name:
            raise HTTPException(status_code=400, detail="run name is required")
        return await _run_sync(handlers.update_run_payload, store, run_id, name)

    @app.get("/api/runs/{run_id}/metrics")
    async def run_metrics(
        run_id: Annotated[str, Path(min_length=1)],
        mode: Literal["recent", "full"] = Query(default="recent"),
        limit: int = Query(default=240, ge=1, le=2_000),
    ) -> dict[str, list[dict[str, object]]]:
        return await _run_sync(
            handlers.run_metrics_payload,
            store,
            run_id,
            limit=None if mode == "full" else limit,
        )

    @app.get("/api/runs/{run_id}/track-sampling")
    async def run_track_sampling(
        run_id: Annotated[str, Path(min_length=1)],
    ) -> dict[str, object]:
        return await _run_sync(handlers.run_track_sampling_payload, store, run_id)

    @app.get("/api/runs/{run_id}/export")
    async def export_run(
        run_id: Annotated[str, Path(min_length=1)],
    ) -> FileResponse:
        bundle_path = await _run_sync(handlers.export_run_bundle_path, store, run_id)
        return FileResponse(
            bundle_path,
            filename=f"{run_id}.zip",
            media_type="application/zip",
        )

    @app.post("/api/run-imports", status_code=201)
    async def import_run(
        bundle: Annotated[UploadFile, File(description="Run export zip bundle")],
    ) -> dict[str, dict[str, object]]:
        return await _run_sync(handlers.import_run_bundle_payload, store, bundle)

    @app.post("/api/runs/{run_id}/track-sampling/reset")
    async def reset_run_track_sampling(
        run_id: Annotated[str, Path(min_length=1)],
    ) -> dict[str, bool]:
        return await _run_sync(handlers.reset_run_track_sampling_payload, store, run_id)

    @app.post("/api/runs", status_code=201)
    async def launch_run(request: LaunchRunRequest) -> dict[str, dict[str, object]]:
        name = request.name.strip()
        if not name:
            raise HTTPException(status_code=400, detail="run name is required")
        handlers.validate_source_fields(
            source_run_id=request.source_run_id,
            source_artifact=request.source_artifact,
        )
        return await _run_sync(handlers.launch_run_payload, store, launcher, request, name)

    @app.post("/api/runs/{run_id}/fork", status_code=201)
    async def fork_run(
        run_id: Annotated[str, Path(min_length=1)],
        request: ForkRunRequest,
    ) -> dict[str, dict[str, object]]:
        return await _run_sync(handlers.fork_run_payload, store, launcher, run_id, request)

    @app.post("/api/runs/{run_id}/pause")
    async def pause_run(
        run_id: Annotated[str, Path(min_length=1)],
    ) -> dict[str, dict[str, object]]:
        return await _run_sync(handlers.pause_run_payload, store, launcher, run_id)

    @app.post("/api/runs/{run_id}/stop")
    async def stop_run(
        run_id: Annotated[str, Path(min_length=1)],
    ) -> dict[str, dict[str, object]]:
        return await _run_sync(handlers.stop_run_payload, store, launcher, run_id)

    @app.post("/api/runs/{run_id}/resume")
    async def resume_run(
        run_id: Annotated[str, Path(min_length=1)],
    ) -> dict[str, dict[str, object]]:
        return await _run_sync(handlers.resume_run_payload, store, launcher, run_id)

    @app.delete("/api/runs/{run_id}")
    async def delete_run(run_id: Annotated[str, Path(min_length=1)]) -> dict[str, bool]:
        return await _run_sync(handlers.delete_run_payload, store, run_id)

    @app.delete("/api/lineages/{lineage_id}")
    async def delete_lineage(lineage_id: Annotated[str, Path(min_length=1)]) -> dict[str, bool]:
        return await _run_sync(handlers.delete_lineage_payload, store, lineage_id)

    @app.put("/api/lineages/{lineage_id}/groups")
    async def update_lineage_groups(
        lineage_id: Annotated[str, Path(min_length=1)],
        request: UpdateLineageGroupsRequest,
    ) -> dict[str, object]:
        return await _run_sync(handlers.update_lineage_groups_payload, store, lineage_id, request)

    @app.post("/api/tensorboard-views/rebuild")
    async def rebuild_tensorboard_views_endpoint() -> dict[str, object]:
        return await _run_sync(handlers.rebuild_tensorboard_views_payload, store)

    @app.post("/api/runs/{run_id}/open-dir")
    async def open_run_dir(run_id: Annotated[str, Path(min_length=1)]) -> dict[str, bool]:
        return await _run_sync(handlers.open_run_dir_payload, store, run_id)

    @app.post("/api/runs/{run_id}/watch")
    async def watch_run(
        run_id: Annotated[str, Path(min_length=1)],
        request: WatchRunRequest | None = None,
        artifact: str = Query(default="latest"),
    ) -> dict[str, str]:
        device: Literal["cpu", "cuda"] = "cuda" if request is None else request.device
        renderer: WatchRenderer | None = None if request is None else request.renderer
        return await _run_sync(
            handlers.watch_run_payload,
            launcher,
            run_id,
            artifact,
            device,
            renderer,
        )

    @app.get("/api/config-metadata")
    async def config_metadata() -> dict[str, object]:
        return await _run_sync(handlers.config_metadata_payload)

    @app.post("/api/policy-preview")
    async def policy_preview(config: ManagedRunConfig) -> dict[str, object]:
        return await _run_sync(handlers.policy_preview_payload, config)

    @app.post("/api/drafts", status_code=201)
    async def create_draft(request: CreateDraftRequest) -> dict[str, dict[str, object]]:
        name = request.name.strip()
        if not name:
            raise HTTPException(status_code=400, detail="draft name is required")
        handlers.validate_source_fields(
            source_run_id=request.source_run_id,
            source_artifact=request.source_artifact,
        )
        return await _run_sync(handlers.create_draft_payload, store, request, name)

    @app.put("/api/drafts/{draft_id}")
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
        return await _run_sync(handlers.update_draft_payload, store, draft_id, request, name)

    @app.delete("/api/drafts/{draft_id}")
    async def delete_draft(
        draft_id: Annotated[str, Path(min_length=1)],
    ) -> dict[str, bool]:
        return await _run_sync(handlers.delete_draft_payload, store, draft_id)

    return app


async def _run_sync(function: Callable[_P, _T], *args: _P.args, **kwargs: _P.kwargs) -> _T:
    return await asyncio.to_thread(partial(function, *args, **kwargs))
