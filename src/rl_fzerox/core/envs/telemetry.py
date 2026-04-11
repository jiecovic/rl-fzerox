# src/rl_fzerox/core/envs/telemetry.py
from __future__ import annotations

from fzerox_emulator import FZeroXTelemetry


def telemetry_boost_active(telemetry: FZeroXTelemetry | None) -> bool:
    """Return whether any boost effect is currently active for the player."""

    if telemetry is None:
        return False
    player = telemetry.player
    return player.boost_timer != 0 or player.dash_pad_boost
