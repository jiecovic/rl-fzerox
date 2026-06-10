# src/rl_fzerox/apps/run_manager/api/routes.py
from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.requests import Request

from rl_fzerox.apps.run_manager.api import handlers
from rl_fzerox.apps.run_manager.api.contracts import RunLauncher
from rl_fzerox.apps.run_manager.api.execution import run_sync as _run_sync
from rl_fzerox.apps.run_manager.api.live import (
    KeyedLiveSnapshotBroadcaster,
    LiveMessageTypes,
    LiveSnapshotBroadcaster,
)
from rl_fzerox.apps.run_manager.api.routers import (
    create_drafts_router,
    create_lineages_router,
    create_metrics_router,
    create_runs_router,
    create_save_games_router,
    create_streams_router,
    create_system_router,
    create_transfer_router,
)
from rl_fzerox.apps.run_manager.launch import ManagerRunLauncher
from rl_fzerox.core.manager import ManagerStore


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
    _register_exception_handlers(app)
    live_broadcaster = LiveSnapshotBroadcaster(
        lambda: _run_sync(handlers.runs_payload, store),
        message_types=LiveMessageTypes(snapshot="runs_snapshot", error="runs_error"),
        error_log_message="failed to poll live run snapshot",
    )
    track_sampling_live_broadcaster = KeyedLiveSnapshotBroadcaster(
        lambda run_id: _run_sync(handlers.run_track_sampling_payload, store, run_id),
        message_types=LiveMessageTypes(
            snapshot="track_sampling_snapshot",
            error="track_sampling_error",
        ),
        error_log_message="failed to poll live track-pool snapshot",
    )

    app.include_router(create_system_router(store))
    app.include_router(create_drafts_router(store))
    app.include_router(create_save_games_router(store, launcher))
    app.include_router(
        create_streams_router(
            live_broadcaster=live_broadcaster,
            track_sampling_live_broadcaster=track_sampling_live_broadcaster,
        )
    )
    app.include_router(create_metrics_router(store))
    app.include_router(create_transfer_router(store))
    app.include_router(create_lineages_router(store))
    app.include_router(create_runs_router(store, launcher))
    return app


def _register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(HTTPException)
    async def handle_http_exception(_request: Request, exc: HTTPException) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})

    @app.exception_handler(RequestValidationError)
    async def handle_validation_exception(
        _request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        return JSONResponse(status_code=400, content={"error": jsonable_encoder(exc.errors())})
