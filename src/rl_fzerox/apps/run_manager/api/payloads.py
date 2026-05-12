# src/rl_fzerox/apps/run_manager/api/payloads.py
from __future__ import annotations

from rl_fzerox.core.manager import (
    ManagedRun,
    ManagedRunDraft,
    ManagedRunEvent,
    ManagedRunMetricSample,
    ManagedRunSummary,
    ManagedRunTemplate,
)
from rl_fzerox.core.manager.artifacts.tensorboard_views import TensorboardViewGroup
from rl_fzerox.core.training.session.callbacks.track_sampling import (
    TrackSamplingRuntimeEntry,
    TrackSamplingRuntimeState,
)


def template_payload(template: ManagedRunTemplate) -> dict[str, object]:
    return {
        "id": template.id,
        "name": template.name,
        "created_at": template.created_at,
        "updated_at": template.updated_at,
        "config": template.config.model_dump(mode="json"),
    }


def draft_payload(draft: ManagedRunDraft) -> dict[str, object]:
    return {
        "id": draft.id,
        "name": draft.name,
        "source_run_id": draft.source_run_id,
        "source_artifact": draft.source_artifact,
        "source_num_timesteps": draft.source_num_timesteps,
        "created_at": draft.created_at,
        "updated_at": draft.updated_at,
        "config": draft.config.model_dump(mode="json"),
    }


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


def tensorboard_view_group_payload(group: TensorboardViewGroup) -> dict[str, object]:
    return {
        "name": group.name,
        "slug": group.slug,
        "path": str(group.path),
        "lineage_count": group.lineage_count,
        "run_count": group.run_count,
    }


def run_metric_payload(sample: ManagedRunMetricSample) -> dict[str, object]:
    return {
        "run_id": sample.run_id,
        "created_at": sample.created_at,
        "total_timesteps": sample.total_timesteps,
        "num_timesteps": sample.num_timesteps,
        "lineage_num_timesteps": sample.lineage_num_timesteps,
        "progress_fraction": sample.progress_fraction,
        "metrics": sample.metrics,
        "fps": sample.fps,
        "episode_reward_mean": sample.episode_reward_mean,
        "episode_length_mean": sample.episode_length_mean,
        "approx_kl": sample.approx_kl,
        "entropy_loss": sample.entropy_loss,
        "value_loss": sample.value_loss,
        "policy_gradient_loss": sample.policy_gradient_loss,
    }


def track_sampling_state_payload(state: TrackSamplingRuntimeState) -> dict[str, object]:
    total_weight = sum(entry.current_weight for entry in state.entries)
    target_step_shares = _target_step_shares(state)
    total_episodes = sum(entry.episode_count for entry in state.entries)
    total_frames = sum(entry.completed_frames for entry in state.entries)
    return {
        "sampling_mode": state.sampling_mode,
        "action_repeat": state.action_repeat,
        "update_episodes": state.update_episodes,
        "ema_alpha": state.ema_alpha,
        "max_weight_scale": state.max_weight_scale,
        "adaptive_completion_weight": state.adaptive_completion_weight,
        "adaptive_target_completion": state.adaptive_target_completion,
        "update_count": state.update_count,
        "episodes_since_update": state.episodes_since_update,
        "entries": [
            {
                "track_id": entry.track_id,
                "course_key": entry.course_key,
                "label": entry.label,
                "current_weight": entry.current_weight,
                "current_probability": (
                    0.0 if total_weight <= 0.0 else entry.current_weight / total_weight
                ),
                "episode_count": entry.episode_count,
                "finished_episode_count": entry.finished_episode_count,
                "success_sample_count": entry.success_sample_count,
                "episode_share": (
                    0.0 if total_episodes <= 0 else entry.episode_count / total_episodes
                ),
                "success_rate": (
                    None
                    if entry.success_sample_count <= 0
                    else entry.finished_episode_count / entry.success_sample_count
                ),
                "target_step_share": target_step_shares.get(entry.course_key, 0.0),
                "completed_frames": entry.completed_frames,
                "completed_env_steps": (
                    0 if state.action_repeat <= 0 else entry.completed_frames // state.action_repeat
                ),
                "step_share": (0.0 if total_frames <= 0 else entry.completed_frames / total_frames),
                "ema_episode_frames": entry.ema_episode_frames,
                "ema_completion_fraction": entry.ema_completion_fraction,
            }
            for entry in state.entries
        ],
    }


def _target_step_shares(state: TrackSamplingRuntimeState) -> dict[str, float]:
    raw_targets = {
        entry.course_key: max(0.0, float(entry.base_weight)) * _target_step_bonus(state, entry)
        for entry in state.entries
    }
    total_target = sum(raw_targets.values())
    if total_target <= 0.0:
        return {entry.course_key: 0.0 for entry in state.entries}
    return {
        course_key: raw_target / total_target for course_key, raw_target in raw_targets.items()
    }


def _target_step_bonus(
    state: TrackSamplingRuntimeState,
    entry: TrackSamplingRuntimeEntry,
) -> float:
    if state.sampling_mode != "adaptive_step_balanced":
        return 1.0
    completion = entry.ema_completion_fraction
    if completion is None:
        return 1.0
    completion_gap = max(state.adaptive_target_completion - completion, 0.0)
    return 1.0 + state.adaptive_completion_weight * completion_gap
