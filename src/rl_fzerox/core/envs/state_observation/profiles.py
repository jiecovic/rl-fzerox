# src/rl_fzerox/core/envs/state_observation/profiles.py
from __future__ import annotations

from collections.abc import Mapping

from fzerox_emulator import FZeroXTelemetry
from rl_fzerox.core.envs.state_observation.types import (
    ObservationStateProfile,
    StateFeature,
    StateVectorSpec,
)
from rl_fzerox.core.envs.state_observation.utils import clamp
from rl_fzerox.core.envs.telemetry import telemetry_boost_active

DEFAULT_STATE_VECTOR_SPEC = StateVectorSpec(
    features=(
        StateFeature("speed_norm", 2.0),
        StateFeature("energy_frac", 1.0),
        StateFeature("reverse_active", 1.0),
        StateFeature("airborne", 1.0),
        StateFeature("can_boost", 1.0),
        StateFeature("boost_active", 1.0),
        StateFeature("left_lean_held", 1.0),
        StateFeature("right_lean_held", 1.0),
        StateFeature("left_press_age_norm", 1.0),
        StateFeature("right_press_age_norm", 1.0),
        StateFeature("recent_boost_pressure", 1.0),
    ),
    speed_normalizer_kph=1_500.0,
    # Mirrors the game's lean double-tap timer window used for side attacks.
    lean_tap_guard_frames=15,
    recent_boost_window_frames=120,
    recent_steer_window_frames=30,
)

RACE_CORE_STATE_VECTOR_SPEC = StateVectorSpec(
    features=(
        StateFeature("speed_norm", 2.0),
        StateFeature("energy_frac", 1.0),
        StateFeature("reverse_active", 1.0),
        StateFeature("airborne", 1.0),
        StateFeature("can_boost", 1.0),
        StateFeature("boost_active", 1.0),
    ),
    speed_normalizer_kph=DEFAULT_STATE_VECTOR_SPEC.speed_normalizer_kph,
    lean_tap_guard_frames=DEFAULT_STATE_VECTOR_SPEC.lean_tap_guard_frames,
    recent_boost_window_frames=DEFAULT_STATE_VECTOR_SPEC.recent_boost_window_frames,
    recent_steer_window_frames=DEFAULT_STATE_VECTOR_SPEC.recent_steer_window_frames,
)

STEER_HISTORY_STATE_VECTOR_SPEC = StateVectorSpec(
    features=(
        *DEFAULT_STATE_VECTOR_SPEC.features,
        StateFeature("steer_left_held", 1.0),
        StateFeature("steer_right_held", 1.0),
        # Signed average steering axis over a short window: -1 left, +1 right.
        StateFeature("recent_steer_pressure", 1.0, low=-1.0),
    ),
    speed_normalizer_kph=DEFAULT_STATE_VECTOR_SPEC.speed_normalizer_kph,
    lean_tap_guard_frames=DEFAULT_STATE_VECTOR_SPEC.lean_tap_guard_frames,
    recent_boost_window_frames=DEFAULT_STATE_VECTOR_SPEC.recent_boost_window_frames,
    recent_steer_window_frames=DEFAULT_STATE_VECTOR_SPEC.recent_steer_window_frames,
)

STATE_VECTOR_SPECS: dict[ObservationStateProfile, StateVectorSpec] = {
    "default": DEFAULT_STATE_VECTOR_SPEC,
    "steer_history": STEER_HISTORY_STATE_VECTOR_SPEC,
    "race_core": RACE_CORE_STATE_VECTOR_SPEC,
}


def state_profile_name(value: str) -> ObservationStateProfile:
    if value == "default":
        return "default"
    if value == "steer_history":
        return "steer_history"
    if value == "race_core":
        return "race_core"
    raise ValueError(f"Unsupported observation state profile: {value!r}")


def state_profile_values(
    telemetry: FZeroXTelemetry | None,
    *,
    profile: ObservationStateProfile,
    profile_fields: Mapping[str, float],
) -> list[float]:
    left_held = clamp(float(profile_fields.get("left_lean_held", 0.0)), 0.0, 1.0)
    right_held = clamp(float(profile_fields.get("right_lean_held", 0.0)), 0.0, 1.0)
    left_age = clamp(float(profile_fields.get("left_press_age_norm", 1.0)), 0.0, 1.0)
    right_age = clamp(float(profile_fields.get("right_press_age_norm", 1.0)), 0.0, 1.0)
    boost_pressure = clamp(float(profile_fields.get("recent_boost_pressure", 0.0)), 0.0, 1.0)
    steer_left = clamp(float(profile_fields.get("steer_left_held", 0.0)), 0.0, 1.0)
    steer_right = clamp(float(profile_fields.get("steer_right_held", 0.0)), 0.0, 1.0)
    steer_pressure = clamp(float(profile_fields.get("recent_steer_pressure", 0.0)), -1.0, 1.0)

    race_core_values = _race_core_values(telemetry, spec=STATE_VECTOR_SPECS[profile])
    if profile == "race_core":
        return race_core_values

    values = [
        *race_core_values,
        left_held,
        right_held,
        left_age,
        right_age,
        boost_pressure,
    ]
    if profile == "steer_history":
        values.extend([steer_left, steer_right, steer_pressure])
    return values


def _race_core_values(
    telemetry: FZeroXTelemetry | None,
    *,
    spec: StateVectorSpec,
) -> list[float]:
    if telemetry is None:
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    player = telemetry.player
    max_energy = float(player.max_energy)
    energy_frac = 0.0 if max_energy <= 0.0 else float(player.energy) / max_energy
    boost_active = 1.0 if telemetry_boost_active(telemetry) else 0.0
    return [
        clamp(float(player.speed_kph) / spec.speed_normalizer_kph, 0.0, 2.0),
        clamp(energy_frac, 0.0, 1.0),
        1.0 if player.reverse_timer > 0 else 0.0,
        1.0 if player.airborne else 0.0,
        1.0 if player.can_boost else 0.0,
        boost_active,
    ]
