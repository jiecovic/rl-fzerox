# src/rl_fzerox/core/envs/rewards/progress.py
from __future__ import annotations

from dataclasses import dataclass

from fzerox_emulator import FZeroXTelemetry, StepStatus, StepSummary
from rl_fzerox.core.envs.laps import completed_race_laps


class EpisodeProgressState:
    """Track episode-relative race distance without assuming track geometry."""

    def __init__(self) -> None:
        self._origin = 0.0
        self._has_origin = False

    def reset(self, telemetry: FZeroXTelemetry | None) -> None:
        if telemetry is None or not telemetry.in_race_mode:
            self._origin = 0.0
            self._has_origin = False
            return
        self._origin = telemetry.player.race_distance
        self._has_origin = True

    def ensure_origin(self, telemetry: FZeroXTelemetry) -> None:
        if self._has_origin or not telemetry.in_race_mode:
            return
        self._origin = telemetry.player.race_distance
        self._has_origin = True

    def relative_distance(self, race_distance: float) -> float:
        if not self._has_origin:
            return 0.0
        return max(race_distance - self._origin, 0.0)


@dataclass
class DamagePenaltyState:
    """Accumulate continuous wall/damage pressure with an optional streak ramp."""

    streak_frames: int = 0

    def reset(self) -> None:
        self.streak_frames = 0

    def penalty(
        self,
        summary: StepSummary,
        *,
        frame_penalty: float,
        streak_ramp_penalty: float,
        streak_cap_frames: int,
    ) -> float:
        damage_frames = max(int(summary.damage_taken_frames), 0)
        if damage_frames <= 0:
            self.streak_frames = 0
            return 0.0

        previous_streak = self.streak_frames
        cap = max(int(streak_cap_frames), 0)
        self.streak_frames += damage_frames
        if cap > 0:
            self.streak_frames = min(self.streak_frames, cap)

        penalty = damage_frames * frame_penalty
        if streak_ramp_penalty == 0.0:
            return penalty

        streak_sum = 0
        for frame_offset in range(1, damage_frames + 1):
            streak = previous_streak + frame_offset
            streak_sum += min(streak, cap) if cap > 0 else streak
        return penalty + (streak_sum * streak_ramp_penalty)


def lap_completion_bonus(
    *,
    lap_number: int,
    total_lap_count: int,
    lap_1_completion_bonus: float,
    lap_2_completion_bonus: float,
    final_lap_completion_bonus: float,
) -> float:
    if lap_number <= 1:
        return lap_1_completion_bonus
    if lap_number >= max(total_lap_count, 1):
        return final_lap_completion_bonus
    return lap_2_completion_bonus


def remaining_lap_count(telemetry: FZeroXTelemetry) -> int:
    return max(telemetry.total_lap_count - completed_race_laps(telemetry), 0)


def remaining_step_penalty(
    *,
    status: StepStatus,
    max_episode_steps: int,
    remaining_step_penalty_per_frame: float,
) -> float:
    remaining_steps = max(max_episode_steps - status.step_count, 0)
    return remaining_steps * remaining_step_penalty_per_frame


def truncation_base_penalty(
    truncation_reason: str,
    *,
    stuck_truncation_base_penalty: float,
    wrong_way_truncation_base_penalty: float,
    progress_stalled_truncation_base_penalty: float,
    timeout_truncation_base_penalty: float,
) -> float:
    if truncation_reason == "stuck":
        return stuck_truncation_base_penalty
    if truncation_reason == "wrong_way":
        return wrong_way_truncation_base_penalty
    if truncation_reason == "progress_stalled":
        return progress_stalled_truncation_base_penalty
    if truncation_reason == "timeout":
        return timeout_truncation_base_penalty
    raise ValueError(f"Unsupported truncation reason: {truncation_reason!r}")
