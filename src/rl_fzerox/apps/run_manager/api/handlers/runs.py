# src/rl_fzerox/apps/run_manager/api/handlers/runs.py
from __future__ import annotations

from typing import Literal

from fastapi import HTTPException

from rl_fzerox.apps.run_manager.api.contracts import (
    ForkRunRequest,
    LaunchRunRequest,
    RunLauncher,
)
from rl_fzerox.apps.run_manager.api.handlers.common import require_run, run_response
from rl_fzerox.apps.run_manager.api.payloads import run_summary_payload
from rl_fzerox.apps.run_manager.desktop import open_directory
from rl_fzerox.core.manager import ManagerStore
from rl_fzerox.core.manager.errors import ManagerNameConflictError


def runs_payload(store: ManagerStore) -> dict[str, list[dict[str, object]]]:
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


def update_run_payload(
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
    return run_response(store, run)


def launch_run_payload(
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
    return run_response(store, run)


def fork_run_payload(
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
    return run_response(store, run)


def pause_run_payload(
    store: ManagerStore,
    launcher: RunLauncher,
    run_id: str,
) -> dict[str, dict[str, object]]:
    try:
        run = launcher.request_pause(run_id=run_id)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return run_response(store, run)


def stop_run_payload(
    store: ManagerStore,
    launcher: RunLauncher,
    run_id: str,
) -> dict[str, dict[str, object]]:
    try:
        run = launcher.request_stop(run_id=run_id)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return run_response(store, run)


def resume_run_payload(
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
    return run_response(store, run)


def delete_run_payload(store: ManagerStore, run_id: str) -> dict[str, bool]:
    try:
        deleted = store.delete_run(run_id)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    if not deleted:
        raise HTTPException(status_code=404, detail="run not found")
    store.rebuild_tensorboard_views()
    return {"deleted": True}


def open_run_dir_payload(store: ManagerStore, run_id: str) -> dict[str, bool]:
    run = require_run(store, run_id)
    try:
        open_directory(run.run_dir)
    except RuntimeError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return {"opened": True}


def watch_run_payload(
    launcher: RunLauncher,
    run_id: str,
    artifact: str,
    device: Literal["cpu", "cuda"],
) -> dict[str, str]:
    try:
        status = launcher.watch_artifact(run_id=run_id, artifact=artifact, device=device)
    except FileNotFoundError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return {"status": status}
