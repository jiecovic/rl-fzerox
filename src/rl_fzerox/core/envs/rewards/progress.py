# src/rl_fzerox/core/envs/rewards/progress.py
from __future__ import annotations

from dataclasses import dataclass

from fzerox_emulator import FZeroXTelemetry, StepSummary


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
