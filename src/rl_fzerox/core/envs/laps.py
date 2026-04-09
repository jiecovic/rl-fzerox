# src/rl_fzerox/core/envs/laps.py
from __future__ import annotations

from fzerox_emulator import FZeroXTelemetry


def completed_race_laps(telemetry: FZeroXTelemetry) -> int:
    """Return completed race laps using the same semantics as the in-game HUD."""

    total_lap_count = max(int(telemetry.total_lap_count), 0)
    if telemetry.player.finished:
        return total_lap_count

    # The raw `laps_completed` RAM field increments at the initial start-line
    # crossing while the HUD still shows lap 1/3. Reward/logging follows HUD
    # semantics: current lap 1 means zero completed race laps.
    display_completed_laps = max(int(telemetry.player.lap) - 1, 0)
    return min(display_completed_laps, total_lap_count)
