# src/rl_fzerox/apps/run_manager/api/routes.py
from __future__ import annotations

import asyncio
from collections.abc import Callable
from functools import partial
from typing import Annotated, Literal, TypeVar

from fastapi import FastAPI, HTTPException, Path, Query
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.requests import Request

from rl_fzerox.apps.run_manager.api.contracts import (
    CreateDraftRequest,
    ForkRunRequest,
    LaunchRunRequest,
    RunLauncher,
    UpdateDraftRequest,
    UpdateRunRequest,
)
from rl_fzerox.apps.run_manager.api.payloads import (
    draft_payload,
    run_metric_payload,
    run_payload,
    template_payload,
    track_sampling_state_payload,
)
from rl_fzerox.apps.run_manager.desktop import open_directory
from rl_fzerox.apps.run_manager.launch import ManagerRunLauncher
from rl_fzerox.apps.run_manager.tensorboard_metrics import (
    load_run_metric_samples_from_tensorboard,
)
from rl_fzerox.core.manager import ManagedRun, ManagedRunConfig, ManagerStore
from rl_fzerox.core.manager.architecture import (
    policy_architecture_preview,
    run_manager_config_metadata,
)
from rl_fzerox.core.manager.errors import ManagerNameConflictError
from rl_fzerox.core.training.runs import RUN_LAYOUT
from rl_fzerox.core.training.session.callbacks.track_sampling import (
    load_track_sampling_runtime_state,
)

SyncReturn = TypeVar("SyncReturn")


def create_manager_api_app(
    store: ManagerStore,
    *,
    run_launcher: RunLauncher | None = None,
) -> FastAPI:
    """Create the local REST API app for the run manager."""

    store.initialize()
    app = FastAPI(title="F-Zero X Run Manager", version="0.1.0")
    launcher = run_launcher or ManagerRunLauncher(store)

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
        items = await _run_sync(store.list_templates)
        return {"templates": [template_payload(item) for item in items]}

    @app.get("/api/drafts")
    async def drafts() -> dict[str, list[dict[str, object]]]:
        items = await _run_sync(store.list_drafts)
        return {"drafts": [draft_payload(item) for item in items]}

    @app.get("/api/runs")
    async def runs() -> dict[str, list[dict[str, object]]]:
        visible_runs = await _run_sync(store.list_visible_runs)
        recent_events = await _run_sync(
            store.list_recent_run_events,
            tuple(run.id for run in visible_runs),
            limit_per_run=6,
        )
        return {
            "runs": [
                run_payload(item, recent_events=recent_events.get(item.id, ()))
                for item in visible_runs
            ]
        }

    @app.put("/api/runs/{run_id}")
    async def update_run(
        run_id: Annotated[str, Path(min_length=1)],
        request: UpdateRunRequest,
    ) -> dict[str, dict[str, object]]:
        name = request.name.strip()
        if not name:
            raise HTTPException(status_code=400, detail="run name is required")
        try:
            run = await _run_sync(store.update_run_name, run_id=run_id, name=name)
        except ManagerNameConflictError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        if run is None:
            raise HTTPException(status_code=404, detail="run not found")
        return await _run_response(store, run)

    @app.get("/api/runs/{run_id}/metrics")
    async def run_metrics(
        run_id: Annotated[str, Path(min_length=1)],
        mode: Literal["recent", "full"] = Query(default="recent"),
        limit: int = Query(default=240, ge=1, le=2_000),
    ) -> dict[str, list[dict[str, object]]]:
        run = await _require_run(store, run_id)
        samples = await _run_sync(
            load_run_metric_samples_from_tensorboard,
            run,
            limit=None if mode == "full" else limit,
        )
        return {"samples": [run_metric_payload(item) for item in samples]}

    @app.get("/api/runs/{run_id}/track-sampling")
    async def run_track_sampling(
        run_id: Annotated[str, Path(min_length=1)],
    ) -> dict[str, object]:
        run = await _require_run(store, run_id)
        state = await _run_sync(
            load_track_sampling_runtime_state,
            run.run_dir / RUN_LAYOUT.runtime_dirname / RUN_LAYOUT.track_sampling_state_filename,
        )
        return {"state": None if state is None else track_sampling_state_payload(state)}

    @app.post("/api/runs/{run_id}/track-sampling/reset")
    async def reset_run_track_sampling(
        run_id: Annotated[str, Path(min_length=1)],
    ) -> dict[str, bool]:
        run = await _require_run(store, run_id)
        if run.status != "stopped":
            raise HTTPException(
                status_code=400,
                detail="track-pool stats can only be reset while the run is stopped",
            )
        await _run_sync(_reset_track_sampling_state, store, run)
        return {"reset": True}

    @app.post("/api/runs", status_code=201)
    async def launch_run(request: LaunchRunRequest) -> dict[str, dict[str, object]]:
        name = request.name.strip()
        if not name:
            raise HTTPException(status_code=400, detail="run name is required")
        _validate_source_fields(
            source_run_id=request.source_run_id,
            source_artifact=request.source_artifact,
        )
        if request.source_run_id is not None and request.draft_id is None:
            raise HTTPException(
                status_code=400,
                detail="fork launches must come from a persisted fork draft",
            )
        try:
            run = await _run_sync(
                launcher.launch,
                name=name,
                config=request.config,
                draft_id=request.draft_id,
                source_run_id=request.source_run_id,
                source_artifact=request.source_artifact,
            )
        except ManagerNameConflictError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return await _run_response(store, run)

    @app.post("/api/runs/{run_id}/fork", status_code=201)
    async def fork_run(
        run_id: Annotated[str, Path(min_length=1)],
        request: ForkRunRequest,
    ) -> dict[str, dict[str, object]]:
        try:
            run = await _run_sync(
                launcher.fork,
                run_id=run_id,
                artifact=request.artifact,
                name=request.name,
                config=request.config,
            )
        except ManagerNameConflictError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return await _run_response(store, run)

    @app.post("/api/runs/{run_id}/pause")
    async def pause_run(
        run_id: Annotated[str, Path(min_length=1)],
    ) -> dict[str, dict[str, object]]:
        try:
            run = await _run_sync(launcher.request_pause, run_id=run_id)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return await _run_response(store, run)

    @app.post("/api/runs/{run_id}/stop")
    async def stop_run(
        run_id: Annotated[str, Path(min_length=1)],
    ) -> dict[str, dict[str, object]]:
        try:
            run = await _run_sync(launcher.request_stop, run_id=run_id)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return await _run_response(store, run)

    @app.post("/api/runs/{run_id}/resume")
    async def resume_run(
        run_id: Annotated[str, Path(min_length=1)],
    ) -> dict[str, dict[str, object]]:
        try:
            run = await _run_sync(launcher.resume, run_id=run_id)
        except FileNotFoundError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return await _run_response(store, run)

    @app.delete("/api/runs/{run_id}")
    async def delete_run(run_id: Annotated[str, Path(min_length=1)]) -> dict[str, bool]:
        try:
            deleted = await _run_sync(store.delete_run, run_id)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        if not deleted:
            raise HTTPException(status_code=404, detail="run not found")
        return {"deleted": True}

    @app.delete("/api/lineages/{lineage_id}")
    async def delete_lineage(lineage_id: Annotated[str, Path(min_length=1)]) -> dict[str, bool]:
        try:
            deleted = await _run_sync(store.delete_lineage, lineage_id)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        if not deleted:
            raise HTTPException(status_code=404, detail="lineage not found")
        return {"deleted": True}

    @app.post("/api/runs/{run_id}/open-dir")
    async def open_run_dir(run_id: Annotated[str, Path(min_length=1)]) -> dict[str, bool]:
        run = await _require_run(store, run_id)
        try:
            await _run_sync(open_directory, run.run_dir)
        except RuntimeError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return {"opened": True}

    @app.post("/api/runs/{run_id}/watch")
    async def watch_run(
        run_id: Annotated[str, Path(min_length=1)],
        artifact: str = Query(default="latest"),
    ) -> dict[str, str]:
        try:
            status = await _run_sync(launcher.watch_artifact, run_id=run_id, artifact=artifact)
        except FileNotFoundError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        except RuntimeError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return {"status": status}

    @app.get("/api/config-metadata")
    async def config_metadata() -> dict[str, object]:
        metadata = await _run_sync(run_manager_config_metadata)
        return metadata.model_dump(mode="json")

    @app.post("/api/policy-preview")
    async def policy_preview(config: ManagedRunConfig) -> dict[str, object]:
        preview = await _run_sync(policy_architecture_preview, config)
        return preview.model_dump(mode="json")

    @app.post("/api/drafts", status_code=201)
    async def create_draft(request: CreateDraftRequest) -> dict[str, dict[str, object]]:
        name = request.name.strip()
        if not name:
            raise HTTPException(status_code=400, detail="draft name is required")
        _validate_source_fields(
            source_run_id=request.source_run_id,
            source_artifact=request.source_artifact,
        )
        try:
            draft = await _run_sync(
                store.create_draft,
                name=name,
                config=request.config,
                source_run_id=request.source_run_id,
                source_artifact=request.source_artifact,
            )
        except ManagerNameConflictError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return {"draft": draft_payload(draft)}

    @app.put("/api/drafts/{draft_id}")
    async def update_draft(
        draft_id: Annotated[str, Path(min_length=1)],
        request: UpdateDraftRequest,
    ) -> dict[str, dict[str, object]]:
        name = request.name.strip()
        if not name:
            raise HTTPException(status_code=400, detail="draft name is required")
        _validate_source_fields(
            source_run_id=request.source_run_id,
            source_artifact=request.source_artifact,
        )
        try:
            draft = await _run_sync(
                store.update_draft,
                draft_id=draft_id,
                name=name,
                config=request.config,
                source_run_id=request.source_run_id,
                source_artifact=request.source_artifact,
            )
        except ManagerNameConflictError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        if draft is None:
            raise HTTPException(status_code=404, detail="draft not found")
        return {"draft": draft_payload(draft)}

    @app.delete("/api/drafts/{draft_id}")
    async def delete_draft(
        draft_id: Annotated[str, Path(min_length=1)],
    ) -> dict[str, bool]:
        deleted = await _run_sync(store.delete_draft, draft_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="draft not found")
        return {"deleted": True}

    return app


async def _run_response(store: ManagerStore, run: ManagedRun) -> dict[str, dict[str, object]]:
    recent_events = await _run_sync(store.list_recent_run_events, (run.id,), limit_per_run=6)
    return {"run": run_payload(run, recent_events=recent_events.get(run.id, ()))}


async def _require_run(store: ManagerStore, run_id: str) -> ManagedRun:
    run = await _run_sync(store.get_run, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="run not found")
    return run


def _validate_source_fields(
    *,
    source_run_id: str | None,
    source_artifact: Literal["latest", "best"] | None,
) -> None:
    if source_run_id is None and source_artifact is None:
        return
    if source_run_id is None or source_artifact is None:
        raise HTTPException(
            status_code=400,
            detail="source_run_id and source_artifact must either both be set or both be null",
        )


def _reset_track_sampling_state(store: ManagerStore, run: ManagedRun) -> None:
    state_path = run.run_dir / RUN_LAYOUT.runtime_dirname / RUN_LAYOUT.track_sampling_state_filename
    if state_path.exists():
        state_path.unlink()
    store.append_run_event(
        run_id=run.id,
        kind="track_sampling_reset",
        message="track-pool stats reset from manager",
    )


async def _run_sync(
    function: Callable[..., SyncReturn],
    /,
    *args: object,
    **kwargs: object,
) -> SyncReturn:
    return await asyncio.to_thread(partial(function, *args, **kwargs))
