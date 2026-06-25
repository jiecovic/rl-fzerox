# src/rl_fzerox/core/envs/rewards/reward_main/recovery.py
"""Outside-track recovery distance helpers.

These functions reduce telemetry and repeated-step summaries into one recovery
distance signal. The control reward module decides whether that movement should
be rewarded for a given action context.
"""
from __future__ import annotations

from fzerox_emulator import FZeroXTelemetry, StepSummary
from rl_fzerox.core.envs.track_bounds import (
    telemetry_outside_track_bounds,
    track_recovery_segment_distance,
)


def outside_track_recovery_distance(
    telemetry: FZeroXTelemetry | None,
) -> float | None:
    if telemetry is None:
        return None
    return track_recovery_segment_distance(telemetry.player)


def previous_outside_track_recovery_distance(
    telemetry: FZeroXTelemetry | None,
) -> float | None:
    if telemetry is None or not telemetry_outside_track_bounds(telemetry):
        return None
    return outside_track_recovery_distance(telemetry)


def outside_track_recovery_baseline(
    *,
    previous_distance: float | None,
    current_distance: float | None,
    outside_track_bounds: bool,
    previous_armed: bool,
    current_armed: bool,
) -> float | None:
    if current_distance is None:
        return previous_distance
    if outside_track_bounds and current_armed and not previous_armed:
        return 0.0
    if outside_track_bounds and previous_distance is None:
        return 0.0
    return previous_distance


def outside_track_recovery_airborne_frames(
    *,
    previous_frames: int,
    summary: StepSummary,
    excursion_active: bool,
) -> int:
    if not excursion_active:
        return 0
    current_frames = max(int(summary.airborne_frames), 0)
    if previous_frames <= 0 and current_frames <= 0:
        return 0
    return max(previous_frames, 0) + current_frames


def outside_track_recovery_armed(
    *,
    previous_armed: bool,
    airborne_frames: int,
    grace_frames: int,
) -> bool:
    if previous_armed:
        return True
    if airborne_frames <= 0:
        return False
    return airborne_frames >= max(grace_frames, 0)
