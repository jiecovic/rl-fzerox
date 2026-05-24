# src/rl_fzerox/core/training/session/callbacks/track_sampling/weights.py
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from rl_fzerox.core.training.session.callbacks.track_sampling.state import TrackStepStats


@dataclass(frozen=True, slots=True)
class CourseSamplingWeights:
    target_frame_weight: float
    expected_episode_frames: float
    reset_weight: float
    frame_debt: float


@dataclass(frozen=True, slots=True)
class StepBalanceSchedulerSettings:
    """Internal scheduler blend; avoids pure-debt starvation."""

    steady_state_probability_share: float = 0.2


STEP_BALANCE_SCHEDULER_SETTINGS = StepBalanceSchedulerSettings()


def distribute_course_weight(
    *,
    course_weight: float,
    entry_ids: Sequence[str],
    entry_base_weights: Mapping[str, float],
    total_entry_base_weight: float,
) -> dict[str, float]:
    if total_entry_base_weight > 0.0:
        return {
            entry_id: course_weight * entry_base_weights[entry_id] / total_entry_base_weight
            for entry_id in entry_ids
        }
    if not entry_ids:
        return {}
    equal_weight = course_weight / len(entry_ids)
    return {entry_id: equal_weight for entry_id in entry_ids}


def blend_course_sampling_weights(
    *,
    steady_state_weights: Mapping[str, CourseSamplingWeights],
    debt_weights: Mapping[str, CourseSamplingWeights],
    steady_state_share: float,
) -> dict[str, CourseSamplingWeights]:
    steady_total = sum(weights.reset_weight for weights in steady_state_weights.values())
    debt_total = sum(weights.reset_weight for weights in debt_weights.values())
    if steady_total <= 0.0 or debt_total <= 0.0:
        return dict(steady_state_weights)

    steady_share = max(0.0, min(1.0, steady_state_share))
    debt_share = 1.0 - steady_share
    return {
        course_key: CourseSamplingWeights(
            target_frame_weight=debt_weights[course_key].target_frame_weight,
            expected_episode_frames=debt_weights[course_key].expected_episode_frames,
            reset_weight=(
                steady_share * steady_state_weights[course_key].reset_weight / steady_total
                + debt_share * debt_weights[course_key].reset_weight / debt_total
            ),
            frame_debt=debt_weights[course_key].frame_debt,
        )
        for course_key in debt_weights
    }


def adaptive_difficulty_bonus(
    *,
    sampling_mode: str,
    max_weight_scale: float,
    completion_weight: float,
    target_completion: float,
    update_episodes: int,
    completion_fraction: float | None,
    finished_episode_count: int,
    success_sample_count: int,
) -> float:
    if sampling_mode != "adaptive_step_balanced":
        return 1.0
    if max_weight_scale <= 1.0 or completion_weight <= 0.0 or target_completion <= 0.0:
        return 1.0
    completion_gap = normalized_completion_gap(
        observed_completion=completion_fraction,
        target_completion=target_completion,
    )
    finish_gap = normalized_completion_gap(
        observed_completion=observed_finish_rate(
            finished_episode_count=finished_episode_count,
            success_sample_count=success_sample_count,
        ),
        target_completion=target_completion,
    )
    difficulty_signal = max(
        completion_gap,
        finish_rate_confidence(
            success_sample_count=success_sample_count,
            update_episodes=update_episodes,
        )
        * finish_gap,
    )
    bonus = 1.0 + (max_weight_scale - 1.0) * completion_weight * difficulty_signal
    return min(bonus, max_weight_scale)


def target_frame_debt(
    stats: TrackStepStats,
    *,
    target_frame_weight: float,
    total_target_frame_weight: float,
    total_completed_frames: int,
) -> float:
    if total_target_frame_weight <= 0.0 or total_completed_frames <= 0:
        return 0.0
    target_share = target_frame_weight / total_target_frame_weight
    target_frames = target_share * total_completed_frames
    return max(0.0, target_frames - stats.completed_frames)


def expected_episode_frames(
    stats: TrackStepStats,
    *,
    fallback_frames: float,
) -> float:
    if stats.ema_episode_frames is None:
        return max(1.0, float(fallback_frames))
    return max(1.0, float(stats.ema_episode_frames))


def normalized_completion_gap(
    *,
    observed_completion: float | None,
    target_completion: float,
) -> float:
    if observed_completion is None or target_completion <= 0.0:
        return 0.0
    return max(target_completion - observed_completion, 0.0) / target_completion


def observed_finish_rate(
    *,
    finished_episode_count: int,
    success_sample_count: int,
) -> float | None:
    if success_sample_count <= 0:
        return None
    return max(0.0, min(1.0, finished_episode_count / success_sample_count))


def finish_rate_confidence(
    *,
    success_sample_count: int,
    update_episodes: int,
) -> float:
    confidence_episodes = max(1, int(update_episodes)) * 4
    return max(0.0, min(1.0, success_sample_count / confidence_episodes))
