# src/rl_fzerox/core/envs/rewards/progress.py
"""Episode-relative progress accounting shared by reward profiles.

`EpisodeProgressState` tracks monotonic race-distance deltas from telemetry.
Profile modules decide how to turn that progress into shaped reward and when
to gate or bucket payouts.
"""

from __future__ import annotations

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


def impact_frame_penalty(summary: StepSummary, *, frame_penalty: float) -> float:
    impact_frames = max(int(summary.impact_frames), 0)
    if impact_frames <= 0 or frame_penalty == 0.0:
        return 0.0
    return impact_frames * frame_penalty
