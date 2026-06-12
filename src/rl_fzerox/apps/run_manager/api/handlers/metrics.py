# src/rl_fzerox/apps/run_manager/api/handlers/metrics.py
from __future__ import annotations

from fastapi import HTTPException

from rl_fzerox.apps.run_manager.api.handlers.common import require_run
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
from rl_fzerox.core.engine_tuning import EngineBanditSettings, EngineTuningRuntimeState
from rl_fzerox.core.manager import ManagedRun, ManagerStore
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
        "enabled": run.config.vehicle.engine_mode == "adaptive_bandit",
        "state": None
        if state is None
        else engine_tuning_state_payload(
            state,
            settings=_engine_bandit_settings(run),
        ),
    }


def _run_engine_tuning_state(
    run: ManagedRun,
    *,
    artifact: str,
) -> EngineTuningRuntimeState | None:
    if run.config.vehicle.engine_mode != "adaptive_bandit":
        return None
    try:
        policy_path = resolve_policy_artifact_path(run.run_dir, artifact=artifact)
    except FileNotFoundError:
        return None
    return load_engine_tuning_checkpoint_state(policy_path)


def _engine_bandit_settings(run: ManagedRun) -> EngineBanditSettings:
    vehicle = run.config.vehicle
    return EngineBanditSettings(
        min_raw_value=vehicle.engine_setting_min_raw_value,
        max_raw_value=vehicle.engine_setting_max_raw_value,
        bin_size=vehicle.adaptive_engine_bin_size,
        stat_decay=vehicle.adaptive_engine_stat_decay,
        prior_mean=vehicle.adaptive_engine_prior_mean,
        prior_strength=vehicle.adaptive_engine_prior_strength,
        exploration_scale=vehicle.adaptive_engine_exploration_scale,
        uniform_exploration=vehicle.adaptive_engine_uniform_exploration,
        completion_weight=vehicle.adaptive_engine_completion_weight,
        finish_bonus=vehicle.adaptive_engine_finish_bonus,
        position_weight=vehicle.adaptive_engine_position_weight,
    )


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
    store.clear_run_track_sampling_state(run.id)
    store.append_run_event(
        run_id=run.id,
        kind="track_sampling_reset",
        message="track-pool stats reset from manager",
    )
