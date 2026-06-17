# src/rl_fzerox/core/envs/telemetry.py
from __future__ import annotations

from fzerox_emulator import FZeroXTelemetry

CAN_BOOST_MIN_ENERGY = 30.0


def telemetry_boost_active(telemetry: FZeroXTelemetry | None) -> bool:
    """Return whether any boost effect is currently active for the player."""

    if telemetry is None:
        return False
    player = telemetry.player
    return player.boost_timer != 0 or player.dash_pad_boost


def telemetry_can_boost(
    telemetry: FZeroXTelemetry | None,
    *,
    min_energy: float = CAN_BOOST_MIN_ENERGY,
) -> bool:
    """Return whether manual boost is unlocked and has enough energy to matter.

    F-Zero X exposes a raw post-lap ``can_boost`` bit. In project terms that
    means boost is unlocked; this helper returns whether boost is actually
    available for policy/reward logic by also checking the energy threshold.
    """

    if telemetry is None:
        return False
    player = telemetry.player
    boost_unlocked = bool(player.can_boost)
    if not boost_unlocked:
        return False
    return float(player.energy) >= float(min_energy)
