# src/rl_fzerox/core/envs/observations/state/components/vehicle.py
"""State-vector features for health, speed, boost, and race timing.

These features describe the current vehicle/race status that the policy may
observe. Control history, course identity, and track geometry are handled by
separate components.
"""

from __future__ import annotations

from collections.abc import Mapping

from fzerox_emulator import FZeroXTelemetry
from rl_fzerox.core.domain.observations import ObservationStateComponentSettings
from rl_fzerox.core.envs.observations.state.types import OBSERVATION_STATE_DEFAULTS, StateFeature
from rl_fzerox.core.envs.observations.state.utils import clamp
from rl_fzerox.core.envs.telemetry import telemetry_boost_active, telemetry_can_boost


def vehicle_component_features(
    _component: ObservationStateComponentSettings,
) -> tuple[StateFeature, ...]:
    return vehicle_state_features()


def vehicle_component_values(
    telemetry: FZeroXTelemetry | None,
    _component: ObservationStateComponentSettings,
    _action_history: Mapping[str, float],
) -> list[float]:
    return vehicle_state_values(telemetry)


def vehicle_state_features() -> tuple[StateFeature, ...]:
    return (
        StateFeature("vehicle_state.speed_norm", 2.0),
        StateFeature("vehicle_state.energy_frac", 1.0),
        StateFeature("vehicle_state.reverse_active", 1.0),
        StateFeature("vehicle_state.airborne", 1.0),
        StateFeature("vehicle_state.can_boost", 1.0),
        StateFeature("vehicle_state.boost_active", 1.0),
        StateFeature("vehicle_state.lateral_velocity_norm", 1.0, low=-1.0),
        StateFeature("vehicle_state.sliding_active", 1.0),
    )


def vehicle_state_values(telemetry: FZeroXTelemetry | None) -> list[float]:
    if telemetry is None:
        return [0.0] * len(vehicle_state_features())
    player = telemetry.player
    energy_frac = 0.0 if player.max_energy <= 0.0 else player.energy / player.max_energy
    lateral_velocity = float(player.local_lateral_velocity)
    return [
        clamp(float(player.speed_kph) / OBSERVATION_STATE_DEFAULTS.speed_normalizer_kph, 0.0, 2.0),
        clamp(float(energy_frac), 0.0, 1.0),
        1.0 if player.reverse_timer > 0 else 0.0,
        1.0 if player.airborne else 0.0,
        1.0 if telemetry_can_boost(telemetry) else 0.0,
        1.0 if telemetry_boost_active(telemetry) else 0.0,
        clamp(
            lateral_velocity / OBSERVATION_STATE_DEFAULTS.lateral_velocity_normalizer,
            -1.0,
            1.0,
        ),
        1.0
        if (
            not player.airborne
            and abs(lateral_velocity)
            > OBSERVATION_STATE_DEFAULTS.sliding_lateral_velocity_threshold
        )
        else 0.0,
    ]
