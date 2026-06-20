# src/rl_fzerox/apps/run_manager/api/payloads/track_sampling.py
from __future__ import annotations

from rl_fzerox.core.training.session.callbacks.track_sampling import (
    TrackSamplingRuntimeEntry,
    TrackSamplingRuntimeState,
    adaptive_target_bonus,
)


def track_sampling_state_payload(
    state: TrackSamplingRuntimeState,
) -> dict[str, object]:
    current_probabilities = _current_probabilities(state)
    target_step_shares = _target_step_shares(state)
    scheduler_env_steps = _scheduler_env_steps_by_course(state)
    total_episodes = sum(entry.episode_count for entry in state.entries)
    total_scheduler_env_steps = sum(scheduler_env_steps.values())
    return {
        "sampling_mode": state.sampling_mode,
        "action_repeat": state.action_repeat,
        "update_episodes": state.update_episodes,
        "ema_alpha": state.ema_alpha,
        "max_weight_scale": state.max_weight_scale,
        "adaptive_completion_weight": state.adaptive_completion_weight,
        "adaptive_target_completion": state.adaptive_target_completion,
        "adaptive_min_confidence_episodes": state.adaptive_min_confidence_episodes,
        "adaptive_confidence_scale": state.adaptive_confidence_scale,
        "deficit_budget_difficulty_metric": state.deficit_budget_difficulty_metric,
        "deficit_budget_warmup_min_episodes_per_course": (
            state.deficit_budget_warmup_min_episodes_per_course
        ),
        "update_count": state.update_count,
        "episodes_since_update": state.episodes_since_update,
        "entries": [
            {
                "track_id": entry.track_id,
                "course_key": entry.course_key,
                "label": entry.label,
                "current_weight": entry.current_weight,
                "current_probability": current_probabilities.get(entry.course_key, 0.0),
                "episode_count": entry.episode_count,
                "finished_episode_count": entry.finished_episode_count,
                "success_sample_count": entry.success_sample_count,
                "completion_sample_count": entry.completion_sample_count,
                "completion_fraction_total": entry.completion_fraction_total,
                "episode_share": (
                    0.0 if total_episodes <= 0 else entry.episode_count / total_episodes
                ),
                "completion_rate": _completion_rate(entry),
                "success_rate": (
                    None
                    if entry.success_sample_count <= 0
                    else entry.finished_episode_count / entry.success_sample_count
                ),
                "generation_episode_count": entry.generation_episode_count,
                "generation_finished_episode_count": entry.generation_finished_episode_count,
                "generation_success_sample_count": entry.generation_success_sample_count,
                "generation_success_rate": (
                    None
                    if entry.generation_success_sample_count <= 0
                    else entry.generation_finished_episode_count
                    / entry.generation_success_sample_count
                ),
                "generation_ema_completion_fraction": entry.generation_ema_completion_fraction,
                "target_step_share": target_step_shares.get(entry.course_key, 0.0),
                "completed_frames": entry.completed_frames,
                "completed_env_steps": scheduler_env_steps.get(entry.course_key, 0),
                "measurement_env_steps": _measurement_env_steps(state, entry),
                "step_share": (
                    0.0
                    if total_scheduler_env_steps <= 0
                    else scheduler_env_steps.get(entry.course_key, 0) / total_scheduler_env_steps
                ),
                "ema_episode_frames": entry.ema_episode_frames,
                "ema_completion_fraction": entry.ema_completion_fraction,
                "ema_finish_rate": entry.ema_finish_rate,
                "current_problem_score": entry.current_problem_score,
                "generated_course_slot": entry.generated_course_slot,
                "generated_course_generation": entry.generated_course_generation,
            }
            for entry in state.entries
        ],
    }


def _scheduler_env_steps_by_course(state: TrackSamplingRuntimeState) -> dict[str, int]:
    scheduler = state.deficit_budget_scheduler
    if scheduler is None:
        return {entry.course_key: _measurement_env_steps(state, entry) for entry in state.entries}
    scheduler_steps = {
        entry.course_key: max(0, int(entry.scheduler_env_steps)) for entry in scheduler.entries
    }
    return {
        entry.course_key: scheduler_steps.get(
            entry.course_key,
            _measurement_env_steps(state, entry),
        )
        for entry in state.entries
    }


def _measurement_env_steps(
    state: TrackSamplingRuntimeState,
    entry: TrackSamplingRuntimeEntry,
) -> int:
    return 0 if state.action_repeat <= 0 else entry.completed_frames // state.action_repeat


def _current_probabilities(state: TrackSamplingRuntimeState) -> dict[str, float]:
    if state.sampling_mode == "fixed_env":
        probability = 0.0 if not state.entries else 1.0 / len(state.entries)
        return {entry.course_key: probability for entry in state.entries}
    total_weight = sum(entry.current_weight for entry in state.entries)
    if total_weight <= 0.0:
        return {entry.course_key: 0.0 for entry in state.entries}
    return {entry.course_key: entry.current_weight / total_weight for entry in state.entries}


def _completion_rate(entry: TrackSamplingRuntimeEntry) -> float | None:
    if entry.completion_sample_count <= 0:
        return None
    return max(0.0, min(1.0, entry.completion_fraction_total / entry.completion_sample_count))


def _target_step_shares(state: TrackSamplingRuntimeState) -> dict[str, float]:
    if state.sampling_mode == "fixed_env":
        return {entry.course_key: 0.0 for entry in state.entries}
    if state.sampling_mode == "deficit_budget":
        return _deficit_budget_target_step_shares(state)
    raw_targets = {
        entry.course_key: max(0.0, float(entry.base_weight)) * _target_step_bonus(state, entry)
        for entry in state.entries
    }
    total_target = sum(raw_targets.values())
    if total_target <= 0.0:
        return {entry.course_key: 0.0 for entry in state.entries}
    return {course_key: raw_target / total_target for course_key, raw_target in raw_targets.items()}


def _deficit_budget_target_step_shares(
    state: TrackSamplingRuntimeState,
) -> dict[str, float]:
    if not state.entries:
        return {}
    uniform_share = 1.0 / len(state.entries)
    adaptive_fraction = max(0.0, min(1.0, float(state.adaptive_completion_weight)))
    uniform_fraction = 1.0 - adaptive_fraction
    total_weight = sum(max(0.0, float(entry.current_weight)) for entry in state.entries)
    if total_weight <= 0.0:
        return {entry.course_key: uniform_share for entry in state.entries}
    return {
        entry.course_key: uniform_fraction * uniform_share
        + adaptive_fraction * (max(0.0, float(entry.current_weight)) / total_weight)
        for entry in state.entries
    }


def _target_step_bonus(
    state: TrackSamplingRuntimeState,
    entry: TrackSamplingRuntimeEntry,
) -> float:
    return adaptive_target_bonus(
        sampling_mode=state.sampling_mode,
        max_weight_scale=state.max_weight_scale,
        completion_weight=state.adaptive_completion_weight,
        target_completion=state.adaptive_target_completion,
        update_episodes=state.update_episodes,
        min_confidence_episodes=state.adaptive_min_confidence_episodes,
        confidence_scale=state.adaptive_confidence_scale,
        completion_fraction=entry.ema_completion_fraction,
        finished_episode_count=entry.finished_episode_count,
        success_sample_count=entry.success_sample_count,
    )
