# src/rl_fzerox/core/envs/rewards/reward_main/airborne.py
"""Airborne landing helpers for `reward_main`.

These helpers summarize jump airtime and peak height when the machine lands.
The tracker owns the previous airborne state; this module only computes the
landing measurements used by reward terms and debug output.
"""
from __future__ import annotations

from fzerox_emulator import FZeroXTelemetry, StepSummary


def landing_airborne_frames(
    *,
    previous_frames: int,
    previous_airborne: bool,
    current_airborne: bool,
    summary: StepSummary,
) -> int:
    current_frames = max(int(summary.airborne_frames), 0)
    if not previous_airborne and not current_airborne and current_frames <= 0:
        return 0
    return max(previous_frames, 0) + current_frames


def landing_airborne_peak_height(
    *,
    previous_peak_height: float,
    telemetry: FZeroXTelemetry | None,
) -> float:
    if telemetry is None or not telemetry.player.airborne:
        return max(float(previous_peak_height), 0.0)
    current_height = max(float(telemetry.player.height_above_ground), 0.0)
    return max(float(previous_peak_height), current_height)
