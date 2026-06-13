# src/rl_fzerox/apps/run_manager/api/handlers/metrics.py
from __future__ import annotations

from fastapi import HTTPException

from rl_fzerox.apps.run_manager.api.handlers.common import require_run, run_response
from rl_fzerox.apps.run_manager.api.payloads.engine_tuning import (
    engine_tuning_state_payload,
)
from rl_fzerox.apps.run_manager.api.payloads.metrics import run_metric_payload
from rl_fzerox.apps.run_manager.api.payloads.track_sampling import (
    track_sampling_state_payload,
)
from rl_fzerox.apps.run_manager.tensorboard_metrics import (
    load_run_metric_samples_from_tensorboard,
)
from rl_fzerox.core.engine_tuning import EngineTunerSettings, EngineTuningRuntimeState
from rl_fzerox.core.engine_tuning.config import engine_tuner_settings
from rl_fzerox.core.manager import ManagedRun, ManagerStore
from rl_fzerox.core.manager.projection.engine_tuning import adaptive_engine_tuning_config
from rl_fzerox.core.training.runs import resolve_policy_artifact_path
from rl_fzerox.core.training.session.artifacts import load_engine_tuning_checkpoint_state


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
    require_run(store, run_id)
    state = store.get_run_track_sampling_state(run_id)
    return {"state": None if state is None else track_sampling_state_payload(state)}


def run_engine_tuning_payload(
    store: ManagerStore,
    run_id: str,
    *,
    artifact: str = "latest",
) -> dict[str, object]:
    run = require_run(store, run_id)
    state = _run_engine_tuning_state(run, artifact=artifact)
    return {
        "enabled": run.config.vehicle.engine_mode == "adaptive_tuner",
        "state": None
        if state is None
        else engine_tuning_state_payload(
            state,
            settings=_engine_tuner_settings(run),
        ),
    }


def _run_engine_tuning_state(
    run: ManagedRun,
    *,
    artifact: str,
) -> EngineTuningRuntimeState | None:
    if run.config.vehicle.engine_mode != "adaptive_tuner":
        return None
    try:
        policy_path = resolve_policy_artifact_path(run.run_dir, artifact=artifact)
    except FileNotFoundError:
        return None
    return load_engine_tuning_checkpoint_state(policy_path)


def _engine_tuner_settings(run: ManagedRun) -> EngineTunerSettings:
    return engine_tuner_settings(adaptive_engine_tuning_config(run.config))


def reset_run_track_sampling_payload(store: ManagerStore, run_id: str) -> dict[str, bool]:
    run = require_run(store, run_id)
    if run.status != "stopped":
        raise HTTPException(
            status_code=400,
            detail="track-pool stats can only be reset while the run is stopped",
        )
    _reset_track_sampling_state(store, run)
    return {"reset": True}


def clear_run_alt_baselines_payload(store: ManagerStore, run_id: str) -> dict[str, object]:
    run = require_run(store, run_id)
    cleared = store.clear_run_alt_baselines(run.id)
    if cleared > 0:
        store.append_run_event(
            run_id=run.id,
            kind="alt_baselines_cleared",
            message=f"cleared {cleared} alt baseline{'s' if cleared != 1 else ''} from manager",
        )
    return {"cleared": cleared, **run_response(store, run)}


def _reset_track_sampling_state(store: ManagerStore, run: ManagedRun) -> None:
    store.clear_run_track_sampling_state(run.id)
    store.append_run_event(
        run_id=run.id,
        kind="track_sampling_reset",
        message="track-pool stats reset from manager",
    )
