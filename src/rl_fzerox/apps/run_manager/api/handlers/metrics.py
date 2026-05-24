# src/rl_fzerox/apps/run_manager/api/handlers/metrics.py
from __future__ import annotations

from fastapi import HTTPException

from rl_fzerox.apps.run_manager.api.handlers.common import require_run
from rl_fzerox.apps.run_manager.api.payloads import (
    run_metric_payload,
    track_sampling_state_payload,
)
from rl_fzerox.apps.run_manager.tensorboard_metrics import (
    load_run_metric_samples_from_tensorboard,
)
from rl_fzerox.core.manager import ManagedRun, ManagerStore
from rl_fzerox.core.training.runs import RUN_LAYOUT
from rl_fzerox.core.training.session.callbacks.track_sampling import (
    load_track_sampling_runtime_state,
)


def run_metrics_payload(
    store: ManagerStore,
    run_id: str,
    *,
    limit: int | None,
) -> dict[str, list[dict[str, object]]]:
    run = require_run(store, run_id)
    samples = load_run_metric_samples_from_tensorboard(run, limit=limit)
    return {"samples": [run_metric_payload(item) for item in samples]}


def run_track_sampling_payload(store: ManagerStore, run_id: str) -> dict[str, object]:
    run = require_run(store, run_id)
    state = load_track_sampling_runtime_state(
        run.run_dir / RUN_LAYOUT.runtime_dirname / RUN_LAYOUT.track_sampling_state_filename,
    )
    return {"state": None if state is None else track_sampling_state_payload(state)}


def reset_run_track_sampling_payload(store: ManagerStore, run_id: str) -> dict[str, bool]:
    run = require_run(store, run_id)
    if run.status != "stopped":
        raise HTTPException(
            status_code=400,
            detail="track-pool stats can only be reset while the run is stopped",
        )
    _reset_track_sampling_state(store, run)
    return {"reset": True}


def _reset_track_sampling_state(store: ManagerStore, run: ManagedRun) -> None:
    state_path = run.run_dir / RUN_LAYOUT.runtime_dirname / RUN_LAYOUT.track_sampling_state_filename
    if state_path.exists():
        state_path.unlink()
    store.append_run_event(
        run_id=run.id,
        kind="track_sampling_reset",
        message="track-pool stats reset from manager",
    )
