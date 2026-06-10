# src/rl_fzerox/apps/run_manager/api/payloads/runs.py
from __future__ import annotations

from rl_fzerox.core.manager import ManagedRun, ManagedRunEvent, ManagedRunSummary


def run_payload(
    run: ManagedRun,
    *,
    recent_events: tuple[ManagedRunEvent, ...] = (),
) -> dict[str, object]:
    payload = run_summary_payload(run, recent_events=recent_events)
    payload["config"] = run.config.model_dump(mode="json")
    return payload


def run_summary_payload(
    run: ManagedRun | ManagedRunSummary,
    *,
    recent_events: tuple[ManagedRunEvent, ...] = (),
    action_repeat: int | None = None,
) -> dict[str, object]:
    resolved_action_repeat = (
        action_repeat
        if action_repeat is not None
        else run.action_repeat
        if isinstance(run, ManagedRunSummary)
        else run.config.action.action_repeat
    )
    return {
        "id": run.id,
        "name": run.name,
        "status": run.status,
        "config_hash": run.config_hash,
        "action_repeat": resolved_action_repeat,
        "vehicle_setup": _run_vehicle_setup_payload(run),
        "created_at": run.created_at,
        "lineage_id": run.lineage_id,
        "lineage_groups": list(run.lineage_groups),
        "lineage_step_offset": run.lineage_step_offset,
        "started_at": run.started_at,
        "stopped_at": run.stopped_at,
        "parent_run_id": run.parent_run_id,
        "source_run_id": run.source_run_id,
        "source_artifact": run.source_artifact,
        "source_num_timesteps": run.source_num_timesteps,
        "pending_command": run.pending_command,
        "worker_heartbeat_at": run.worker_heartbeat_at,
        "runtime": None
        if run.runtime is None
        else {
            "total_timesteps": run.runtime.total_timesteps,
            "num_timesteps": run.runtime.num_timesteps,
            "progress_fraction": run.runtime.progress_fraction,
            "updated_at": run.runtime.updated_at,
            "fps": run.runtime.fps,
            "episode_reward_mean": run.runtime.episode_reward_mean,
            "episode_length_mean": run.runtime.episode_length_mean,
            "approx_kl": run.runtime.approx_kl,
            "entropy_loss": run.runtime.entropy_loss,
            "value_loss": run.runtime.value_loss,
            "policy_gradient_loss": run.runtime.policy_gradient_loss,
        },
        "recent_events": [
            {
                "created_at": event.created_at,
                "kind": event.kind,
                "message": event.message,
            }
            for event in recent_events
        ],
    }


def _run_vehicle_setup_payload(run: ManagedRun | ManagedRunSummary) -> dict[str, object]:
    if isinstance(run, ManagedRunSummary):
        vehicle = run.vehicle_setup
        return {
            "selection_mode": vehicle.selection_mode,
            "selected_vehicle_ids": list(vehicle.selected_vehicle_ids),
            "engine_mode": vehicle.engine_mode,
            "engine_setting_raw_value": vehicle.engine_setting_raw_value,
            "engine_setting_min_raw_value": vehicle.engine_setting_min_raw_value,
            "engine_setting_max_raw_value": vehicle.engine_setting_max_raw_value,
        }
    vehicle = run.config.vehicle
    return {
        "selection_mode": vehicle.selection_mode,
        "selected_vehicle_ids": list(vehicle.selected_vehicle_ids),
        "engine_mode": vehicle.engine_mode,
        "engine_setting_raw_value": vehicle.engine_setting_raw_value,
        "engine_setting_min_raw_value": vehicle.engine_setting_min_raw_value,
        "engine_setting_max_raw_value": vehicle.engine_setting_max_raw_value,
    }
