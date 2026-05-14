# src/rl_fzerox/apps/run_manager/api/routes.py
from __future__ import annotations

import asyncio
from collections.abc import Callable
from functools import partial
from typing import Annotated, Literal, ParamSpec, TypeVar

from fastapi import FastAPI, HTTPException, Path, Query, WebSocket
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
    UpdateLineageGroupsRequest,
    UpdateRunRequest,
)
from rl_fzerox.apps.run_manager.api.live import RunLiveBroadcaster
from rl_fzerox.apps.run_manager.api.payloads import (
    draft_payload,
    run_metric_payload,
    run_payload,
    run_summary_payload,
    template_payload,
    tensorboard_view_group_payload,
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
    live_broadcaster = RunLiveBroadcaster(lambda: _run_sync(_runs_payload, store))

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
        return await _run_sync(_templates_payload, store)

    @app.get("/api/drafts")
    async def drafts() -> dict[str, list[dict[str, object]]]:
        return await _run_sync(_drafts_payload, store)

    @app.get("/api/runs")
    async def runs() -> dict[str, list[dict[str, object]]]:
        return await _run_sync(_runs_payload, store)

    @app.websocket("/api/runs/live")
    async def live_runs(websocket: WebSocket) -> None:
        await live_broadcaster.serve(websocket)

    @app.get("/api/runs/{run_id}")
    async def run_detail(
        run_id: Annotated[str, Path(min_length=1)],
    ) -> dict[str, dict[str, object]]:
        return await _run_sync(_run_response_for_id, store, run_id)

    @app.put("/api/runs/{run_id}")
    async def update_run(
        run_id: Annotated[str, Path(min_length=1)],
        request: UpdateRunRequest,
    ) -> dict[str, dict[str, object]]:
        name = request.name.strip()
        if not name:
            raise HTTPException(status_code=400, detail="run name is required")
        return await _run_sync(_update_run_payload, store, run_id, name)

    @app.get("/api/runs/{run_id}/metrics")
    async def run_metrics(
        run_id: Annotated[str, Path(min_length=1)],
        mode: Literal["recent", "full"] = Query(default="recent"),
        limit: int = Query(default=240, ge=1, le=2_000),
    ) -> dict[str, list[dict[str, object]]]:
        return await _run_sync(
            _run_metrics_payload,
            store,
            run_id,
            limit=None if mode == "full" else limit,
        )

    @app.get("/api/runs/{run_id}/track-sampling")
    async def run_track_sampling(
        run_id: Annotated[str, Path(min_length=1)],
    ) -> dict[str, object]:
        return await _run_sync(_run_track_sampling_payload, store, run_id)

    @app.post("/api/runs/{run_id}/track-sampling/reset")
    async def reset_run_track_sampling(
        run_id: Annotated[str, Path(min_length=1)],
    ) -> dict[str, bool]:
        return await _run_sync(_reset_run_track_sampling_payload, store, run_id)

    @app.post("/api/runs", status_code=201)
    async def launch_run(request: LaunchRunRequest) -> dict[str, dict[str, object]]:
        name = request.name.strip()
        if not name:
            raise HTTPException(status_code=400, detail="run name is required")
        _validate_source_fields(
            source_run_id=request.source_run_id,
            source_artifact=request.source_artifact,
        )
        return await _run_sync(_launch_run_payload, store, launcher, request, name)

    @app.post("/api/runs/{run_id}/fork", status_code=201)
    async def fork_run(
        run_id: Annotated[str, Path(min_length=1)],
        request: ForkRunRequest,
    ) -> dict[str, dict[str, object]]:
        return await _run_sync(_fork_run_payload, store, launcher, run_id, request)

    @app.post("/api/runs/{run_id}/pause")
    async def pause_run(
        run_id: Annotated[str, Path(min_length=1)],
    ) -> dict[str, dict[str, object]]:
        return await _run_sync(_pause_run_payload, store, launcher, run_id)

    @app.post("/api/runs/{run_id}/stop")
    async def stop_run(
        run_id: Annotated[str, Path(min_length=1)],
    ) -> dict[str, dict[str, object]]:
        return await _run_sync(_stop_run_payload, store, launcher, run_id)

    @app.post("/api/runs/{run_id}/resume")
    async def resume_run(
        run_id: Annotated[str, Path(min_length=1)],
    ) -> dict[str, dict[str, object]]:
        return await _run_sync(_resume_run_payload, store, launcher, run_id)

    @app.delete("/api/runs/{run_id}")
    async def delete_run(run_id: Annotated[str, Path(min_length=1)]) -> dict[str, bool]:
        return await _run_sync(_delete_run_payload, store, run_id)

    @app.delete("/api/lineages/{lineage_id}")
    async def delete_lineage(lineage_id: Annotated[str, Path(min_length=1)]) -> dict[str, bool]:
        return await _run_sync(_delete_lineage_payload, store, lineage_id)

    @app.put("/api/lineages/{lineage_id}/groups")
    async def update_lineage_groups(
        lineage_id: Annotated[str, Path(min_length=1)],
        request: UpdateLineageGroupsRequest,
    ) -> dict[str, object]:
        return await _run_sync(_update_lineage_groups_payload, store, lineage_id, request)

    @app.post("/api/tensorboard-views/rebuild")
    async def rebuild_tensorboard_views_endpoint() -> dict[str, object]:
        return await _run_sync(_rebuild_tensorboard_views_payload, store)

    @app.post("/api/runs/{run_id}/open-dir")
    async def open_run_dir(run_id: Annotated[str, Path(min_length=1)]) -> dict[str, bool]:
        return await _run_sync(_open_run_dir_payload, store, run_id)

    @app.post("/api/runs/{run_id}/watch")
    async def watch_run(
        run_id: Annotated[str, Path(min_length=1)],
        artifact: str = Query(default="latest"),
    ) -> dict[str, str]:
        return await _run_sync(_watch_run_payload, launcher, run_id, artifact)

    @app.get("/api/config-metadata")
    async def config_metadata() -> dict[str, object]:
        return await _run_sync(_config_metadata_payload)

    @app.post("/api/policy-preview")
    async def policy_preview(config: ManagedRunConfig) -> dict[str, object]:
        return await _run_sync(_policy_preview_payload, config)

    @app.post("/api/drafts", status_code=201)
    async def create_draft(request: CreateDraftRequest) -> dict[str, dict[str, object]]:
        name = request.name.strip()
        if not name:
            raise HTTPException(status_code=400, detail="draft name is required")
        _validate_source_fields(
            source_run_id=request.source_run_id,
            source_artifact=request.source_artifact,
        )
        return await _run_sync(_create_draft_payload, store, request, name)

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
        return await _run_sync(_update_draft_payload, store, draft_id, request, name)

    @app.delete("/api/drafts/{draft_id}")
    async def delete_draft(
        draft_id: Annotated[str, Path(min_length=1)],
    ) -> dict[str, bool]:
        return await _run_sync(_delete_draft_payload, store, draft_id)

    return app


def _run_response(store: ManagerStore, run: ManagedRun) -> dict[str, dict[str, object]]:
    recent_events = store.list_recent_run_events((run.id,), limit_per_run=6)
    return {"run": run_payload(run, recent_events=recent_events.get(run.id, ()))}


def _templates_payload(store: ManagerStore) -> dict[str, list[dict[str, object]]]:
    items = store.list_templates()
    return {"templates": [template_payload(item) for item in items]}


def _drafts_payload(store: ManagerStore) -> dict[str, list[dict[str, object]]]:
    items = store.list_drafts()
    return {"drafts": [draft_payload(item) for item in items]}


def _runs_payload(store: ManagerStore) -> dict[str, list[dict[str, object]]]:
    visible_runs = store.list_visible_run_summaries()
    recent_events = store.list_recent_run_events(
        tuple(run.id for run in visible_runs),
        limit_per_run=6,
    )
    return {
        "runs": [
            run_summary_payload(item, recent_events=recent_events.get(item.id, ()))
            for item in visible_runs
        ]
    }


def _run_response_for_id(store: ManagerStore, run_id: str) -> dict[str, dict[str, object]]:
    return _run_response(store, _require_run(store, run_id))


def _update_run_payload(
    store: ManagerStore,
    run_id: str,
    name: str,
) -> dict[str, dict[str, object]]:
    try:
        run = store.update_run_name(run_id=run_id, name=name)
    except ManagerNameConflictError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    if run is None:
        raise HTTPException(status_code=404, detail="run not found")
    store.rebuild_tensorboard_views()
    return _run_response(store, run)


def _launch_run_payload(
    store: ManagerStore,
    launcher: RunLauncher,
    request: LaunchRunRequest,
    name: str,
) -> dict[str, dict[str, object]]:
    try:
        run = launcher.launch(
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
    store.rebuild_tensorboard_views()
    return _run_response(store, run)


def _fork_run_payload(
    store: ManagerStore,
    launcher: RunLauncher,
    run_id: str,
    request: ForkRunRequest,
) -> dict[str, dict[str, object]]:
    try:
        run = launcher.fork(
            run_id=run_id,
            artifact=request.artifact,
            name=request.name,
            config=request.config,
        )
    except ManagerNameConflictError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    store.rebuild_tensorboard_views()
    return _run_response(store, run)


def _pause_run_payload(
    store: ManagerStore,
    launcher: RunLauncher,
    run_id: str,
) -> dict[str, dict[str, object]]:
    try:
        run = launcher.request_pause(run_id=run_id)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return _run_response(store, run)


def _stop_run_payload(
    store: ManagerStore,
    launcher: RunLauncher,
    run_id: str,
) -> dict[str, dict[str, object]]:
    try:
        run = launcher.request_stop(run_id=run_id)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return _run_response(store, run)


def _resume_run_payload(
    store: ManagerStore,
    launcher: RunLauncher,
    run_id: str,
) -> dict[str, dict[str, object]]:
    try:
        run = launcher.resume(run_id=run_id)
    except FileNotFoundError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return _run_response(store, run)


def _delete_run_payload(store: ManagerStore, run_id: str) -> dict[str, bool]:
    try:
        deleted = store.delete_run(run_id)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    if not deleted:
        raise HTTPException(status_code=404, detail="run not found")
    store.rebuild_tensorboard_views()
    return {"deleted": True}


def _delete_lineage_payload(store: ManagerStore, lineage_id: str) -> dict[str, bool]:
    try:
        deleted = store.delete_lineage(lineage_id)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    if not deleted:
        raise HTTPException(status_code=404, detail="lineage not found")
    store.rebuild_tensorboard_views()
    return {"deleted": True}


def _update_lineage_groups_payload(
    store: ManagerStore,
    lineage_id: str,
    request: UpdateLineageGroupsRequest,
) -> dict[str, object]:
    try:
        group_names = store.update_lineage_groups(
            lineage_id=lineage_id,
            group_names=request.group_names,
        )
    except ValueError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    view_groups = store.rebuild_tensorboard_views()
    return {
        "lineage_id": lineage_id,
        "lineage_groups": list(group_names),
        "tensorboard_views": [tensorboard_view_group_payload(group) for group in view_groups],
    }


def _rebuild_tensorboard_views_payload(store: ManagerStore) -> dict[str, object]:
    view_groups = store.rebuild_tensorboard_views()
    return {
        "tensorboard_views": [tensorboard_view_group_payload(group) for group in view_groups],
    }


def _run_metrics_payload(
    store: ManagerStore,
    run_id: str,
    *,
    limit: int | None,
) -> dict[str, list[dict[str, object]]]:
    run = _require_run(store, run_id)
    samples = load_run_metric_samples_from_tensorboard(run, limit=limit)
    return {"samples": [run_metric_payload(item) for item in samples]}


def _run_track_sampling_payload(store: ManagerStore, run_id: str) -> dict[str, object]:
    run = _require_run(store, run_id)
    state = load_track_sampling_runtime_state(
        run.run_dir / RUN_LAYOUT.runtime_dirname / RUN_LAYOUT.track_sampling_state_filename,
    )
    return {"state": None if state is None else track_sampling_state_payload(state)}


def _reset_run_track_sampling_payload(store: ManagerStore, run_id: str) -> dict[str, bool]:
    run = _require_run(store, run_id)
    if run.status != "stopped":
        raise HTTPException(
            status_code=400,
            detail="track-pool stats can only be reset while the run is stopped",
        )
    _reset_track_sampling_state(store, run)
    return {"reset": True}


def _open_run_dir_payload(store: ManagerStore, run_id: str) -> dict[str, bool]:
    run = _require_run(store, run_id)
    try:
        open_directory(run.run_dir)
    except RuntimeError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return {"opened": True}


def _watch_run_payload(
    launcher: RunLauncher,
    run_id: str,
    artifact: str,
) -> dict[str, str]:
    try:
        status = launcher.watch_artifact(run_id=run_id, artifact=artifact)
    except FileNotFoundError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return {"status": status}


def _config_metadata_payload() -> dict[str, object]:
    metadata = run_manager_config_metadata()
    return metadata.model_dump(mode="json")


def _policy_preview_payload(config: ManagedRunConfig) -> dict[str, object]:
    preview = policy_architecture_preview(config)
    return preview.model_dump(mode="json")


def _create_draft_payload(
    store: ManagerStore,
    request: CreateDraftRequest,
    name: str,
) -> dict[str, dict[str, object]]:
    try:
        draft = store.create_draft(
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


def _update_draft_payload(
    store: ManagerStore,
    draft_id: str,
    request: UpdateDraftRequest,
    name: str,
) -> dict[str, dict[str, object]]:
    try:
        draft = store.update_draft(
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


def _delete_draft_payload(store: ManagerStore, draft_id: str) -> dict[str, bool]:
    deleted = store.delete_draft(draft_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="draft not found")
    return {"deleted": True}


def _require_run(store: ManagerStore, run_id: str) -> ManagedRun:
    run = store.get_run(run_id)
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


async def _run_sync(function: Callable[_P, _T], *args: _P.args, **kwargs: _P.kwargs) -> _T:
    return await asyncio.to_thread(partial(function, *args, **kwargs))
