# src/rl_fzerox/apps/run_manager/api/payloads.py
from __future__ import annotations

from rl_fzerox.core.manager import (
    ManagedRun,
    ManagedRunDraft,
    ManagedRunEvent,
    ManagedRunMetricSample,
    ManagedRunTemplate,
)
from rl_fzerox.core.training.session.callbacks.track_sampling import TrackSamplingRuntimeState


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
    return {
        "id": run.id,
        "name": run.name,
        "status": run.status,
        "created_at": run.created_at,
        "lineage_id": run.lineage_id,
        "lineage_step_offset": run.lineage_step_offset,
        "started_at": run.started_at,
        "stopped_at": run.stopped_at,
        "parent_run_id": run.parent_run_id,
        "source_run_id": run.source_run_id,
        "source_artifact": run.source_artifact,
        "source_num_timesteps": run.source_num_timesteps,
        "pending_command": run.pending_command,
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
        "config": run.config.model_dump(mode="json"),
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
    total_episodes = sum(entry.episode_count for entry in state.entries)
    total_frames = sum(entry.completed_frames for entry in state.entries)
    return {
        "sampling_mode": state.sampling_mode,
        "action_repeat": state.action_repeat,
        "update_episodes": state.update_episodes,
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
                "completed_frames": entry.completed_frames,
                "completed_env_steps": (
                    0 if state.action_repeat <= 0 else entry.completed_frames // state.action_repeat
                ),
                "step_share": (0.0 if total_frames <= 0 else entry.completed_frames / total_frames),
            }
            for entry in state.entries
        ],
    }
