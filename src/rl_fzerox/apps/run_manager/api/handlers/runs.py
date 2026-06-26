# src/rl_fzerox/apps/run_manager/api/handlers/runs.py
from __future__ import annotations

from typing import Literal

from fastapi import HTTPException

from rl_fzerox.apps.run_manager.api.contracts import (
    ForkRunRequest,
    LaunchRunRequest,
    RunLauncher,
    WatchRenderer,
)
from rl_fzerox.apps.run_manager.api.handlers.common import (
    active_alt_baseline_count,
    require_run,
    run_response,
)
from rl_fzerox.apps.run_manager.api.payloads.runs import run_summary_payload
from rl_fzerox.apps.run_manager.desktop import open_directory
from rl_fzerox.core.manager import ManagedRunSummary, ManagerStore
from rl_fzerox.core.manager.errors import ManagerNameConflictError


def runs_payload(store: ManagerStore) -> dict[str, list[dict[str, object]]]:
    # Run snapshots are the UI polling boundary that advances stale worker
    # leases. Core registry read helpers stay read-only.
    store.reconcile_orphaned_runs()
    return _runs_payload(store, store.list_visible_run_summaries())


def runs_live_payload(store: ManagerStore) -> dict[str, list[dict[str, object]]]:
    """Return live run summaries including installed checkpoint run snapshots."""

    store.reconcile_orphaned_runs()
    runs_by_id = {run.id: run for run in store.list_visible_run_summaries()}
    for checkpoint in store.list_published_checkpoints():
        snapshot_run = store.get_run_summary(checkpoint.run_id)
        if snapshot_run is not None:
            runs_by_id.setdefault(snapshot_run.id, snapshot_run)
    return _runs_payload(store, tuple(runs_by_id.values()))


def _runs_payload(
    store: ManagerStore,
    runs: tuple[ManagedRunSummary, ...],
) -> dict[str, list[dict[str, object]]]:
    recent_events = store.list_recent_run_events(
        tuple(run.id for run in runs),
        limit_per_run=6,
    )
    alt_baseline_counts = {run.id: active_alt_baseline_count(store, run.id) for run in runs}
    return {
        "runs": [
            run_summary_payload(
                item,
                recent_events=recent_events.get(item.id, ()),
                active_alt_baseline_count=alt_baseline_counts.get(item.id, 0),
            )
            for item in runs
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
            source_policy_kind=request.source_policy_kind,
            source_policy_id=request.source_policy_id,
            source_run_id=request.source_run_id,
            source_artifact=request.source_artifact,
            copy_alt_baselines=request.copy_alt_baselines,
            engine_tuning_source_action=request.engine_tuning_source_action,
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
            copy_alt_baselines=request.copy_alt_baselines,
            engine_tuning_source_action=request.engine_tuning_source_action,
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
    renderer: WatchRenderer | None,
    deterministic_policy: bool,
) -> dict[str, str]:
    try:
        status = launcher.watch_artifact(
            run_id=run_id,
            artifact=artifact,
            device=device,
            renderer=renderer,
            deterministic_policy=deterministic_policy,
        )
    except FileNotFoundError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return {"status": status}
