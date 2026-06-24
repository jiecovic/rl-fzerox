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
    """Blend long-term target shares with immediate frame debt.

    Pure debt scheduling can starve a course after one long episode overpays its
    target. A small steady-state share keeps every target reachable while debt
    remains the main correction signal.
    """

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
