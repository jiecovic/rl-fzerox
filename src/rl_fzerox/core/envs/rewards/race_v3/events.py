# src/rl_fzerox/core/envs/rewards/race_v3/events.py
from __future__ import annotations

from fzerox_emulator import FZeroXTelemetry, StepStatus, StepSummary
from rl_fzerox.core.envs.course_effects import CourseEffect, course_effect_raw
from rl_fzerox.core.envs.laps import completed_race_laps
from rl_fzerox.core.envs.rewards.common import finish_placement_bonus
from rl_fzerox.core.envs.rewards.race_v3.weights import RaceV3RewardWeights

_FAILURE_TERMINATION_REASONS = frozenset(
    ("spinning_out", "crashed", "retired", "falling_off_track")
)


class BoostPadRewardTracker:
    """Reward dash-pad entries once per unwrapped progress window."""

    def __init__(self) -> None:
        self._rewarded_progress_windows: set[int] = set()

    @property
    def rewarded_window_count(self) -> int:
        return len(self._rewarded_progress_windows)

    def reset(self) -> None:
        self._rewarded_progress_windows.clear()

    def reward(
        self,
        summary: StepSummary,
        *,
        relative_progress: float,
        weights: RaceV3RewardWeights,
    ) -> float:
        reward = weights.boost_pad_reward
        if reward <= 0.0 or not summary.entered_dash_pad_boost or summary.reverse_active_frames > 0:
            return 0.0
        progress_window = weights.boost_pad_reward_progress_window
        if progress_window <= 0.0:
            return 0.0
        window_index = int(relative_progress // progress_window)
        if window_index in self._rewarded_progress_windows:
            return 0.0
        self._rewarded_progress_windows.add(window_index)
        return reward


class BadSurfaceEntryPenaltyTracker:
    """Penalize transitions into bad course surfaces once per entry."""

    def __init__(self) -> None:
        self._previous_effect = int(CourseEffect.NONE)

    def reset(self, telemetry: FZeroXTelemetry | None = None) -> None:
        self._previous_effect = course_effect_raw(telemetry)

    def penalty(
        self,
        telemetry: FZeroXTelemetry,
        *,
        weights: RaceV3RewardWeights,
        breakdown: dict[str, float],
    ) -> float:
        current_effect = course_effect_raw(telemetry)
        previous_effect = self._previous_effect
        self._previous_effect = current_effect

        if current_effect == previous_effect:
            return 0.0
        if current_effect == CourseEffect.DIRT:
            penalty = weights.dirt_entry_penalty
            label = "dirt_entry"
        elif current_effect == CourseEffect.ICE:
            penalty = weights.ice_entry_penalty
            label = "ice_entry"
        else:
            return 0.0

        if penalty == 0.0:
            return 0.0
        breakdown[label] = penalty
        return penalty


class LapRewardTracker:
    """Pay flat and placement lap rewards exactly once per completed lap."""

    def __init__(self) -> None:
        self._awarded_laps_completed = 0

    @property
    def awarded_laps_completed(self) -> int:
        return self._awarded_laps_completed

    def reset(self, telemetry: FZeroXTelemetry | None) -> None:
        if telemetry is None or not telemetry.in_race_mode:
            self._awarded_laps_completed = 0
            return
        self._awarded_laps_completed = completed_race_laps(telemetry)

    def reward(
        self,
        telemetry: FZeroXTelemetry,
        *,
        weights: RaceV3RewardWeights,
        breakdown: dict[str, float],
    ) -> float:
        race_laps_completed = completed_race_laps(telemetry)
        laps_completed_gain = race_laps_completed - self._awarded_laps_completed
        if laps_completed_gain <= 0:
            return 0.0

        lap_bonus = laps_completed_gain * weights.lap_completion_bonus
        lap_position_bonus = laps_completed_gain * finish_placement_bonus(
            position=telemetry.player.position,
            total_racers=telemetry.total_racers,
            scale=weights.lap_position_scale,
        )

        if lap_bonus:
            breakdown["lap_completion"] = lap_bonus
        if lap_position_bonus:
            breakdown["lap_position"] = lap_position_bonus
        self._awarded_laps_completed = race_laps_completed
        return lap_bonus + lap_position_bonus


def landing_reward(
    *,
    previous_airborne: bool,
    telemetry: FZeroXTelemetry,
    weights: RaceV3RewardWeights,
) -> float:
    reward = weights.airborne_landing_reward
    if reward == 0.0:
        return 0.0
    if not previous_airborne or telemetry.player.airborne:
        return 0.0
    return reward


def terminal_or_truncation_penalty(
    status: StepStatus,
    *,
    weights: RaceV3RewardWeights,
    breakdown: dict[str, float],
) -> float:
    if status.termination_reason == "finished":
        return 0.0
    if status.termination_reason is not None:
        if status.termination_reason not in _FAILURE_TERMINATION_REASONS:
            return 0.0
        penalty = weights.failure_penalty
        breakdown[status.termination_reason] = penalty
        return penalty
    if status.truncation_reason is None:
        return 0.0
    penalty = weights.truncation_penalty
    breakdown[f"{status.truncation_reason}_truncation"] = penalty
    return penalty
