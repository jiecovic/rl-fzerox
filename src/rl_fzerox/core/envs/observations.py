# src/rl_fzerox/core/envs/observations.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

import numpy as np
from gymnasium import spaces

from fzerox_emulator import FZeroXTelemetry, ObservationSpec
from rl_fzerox.core.envs.telemetry import telemetry_boost_active

ObservationMode: TypeAlias = Literal["image", "image_state"]
ObservationStateProfile: TypeAlias = Literal["default", "steer_history"]
ImageObservation: TypeAlias = np.ndarray
ImageStateObservation: TypeAlias = dict[str, np.ndarray]
ObservationValue: TypeAlias = ImageObservation | ImageStateObservation
DEFAULT_OBSERVATION_STATE_PROFILE: ObservationStateProfile = "default"


@dataclass(frozen=True, slots=True)
class StateFeature:
    """One scalar policy-state feature and its observation-space upper bound."""

    name: str
    high: float
    low: float = 0.0


@dataclass(frozen=True, slots=True)
class StateVectorSpec:
    """Ordered scalar state schema appended to image observations."""

    features: tuple[StateFeature, ...]
    speed_normalizer_kph: float
    shoulder_tap_guard_frames: int
    recent_boost_window_frames: int
    recent_steer_window_frames: int

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(feature.name for feature in self.features)

    @property
    def count(self) -> int:
        return len(self.features)

    def high_array(self) -> np.ndarray:
        return np.array([feature.high for feature in self.features], dtype=np.float32)

    def low_array(self) -> np.ndarray:
        return np.array([feature.low for feature in self.features], dtype=np.float32)


DEFAULT_STATE_VECTOR_SPEC = StateVectorSpec(
    features=(
        StateFeature("speed_norm", 2.0),
        StateFeature("energy_frac", 1.0),
        StateFeature("reverse_active", 1.0),
        StateFeature("airborne", 1.0),
        StateFeature("can_boost", 1.0),
        StateFeature("boost_active", 1.0),
        StateFeature("left_drift_held", 1.0),
        StateFeature("right_drift_held", 1.0),
        StateFeature("left_press_age_norm", 1.0),
        StateFeature("right_press_age_norm", 1.0),
        StateFeature("recent_boost_pressure", 1.0),
    ),
    speed_normalizer_kph=1_500.0,
    # Mirrors the game's shoulder double-tap timer window used for side attacks.
    shoulder_tap_guard_frames=15,
    recent_boost_window_frames=120,
    recent_steer_window_frames=30,
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
    shoulder_tap_guard_frames=DEFAULT_STATE_VECTOR_SPEC.shoulder_tap_guard_frames,
    recent_boost_window_frames=DEFAULT_STATE_VECTOR_SPEC.recent_boost_window_frames,
    recent_steer_window_frames=DEFAULT_STATE_VECTOR_SPEC.recent_steer_window_frames,
)
STATE_VECTOR_SPECS: dict[ObservationStateProfile, StateVectorSpec] = {
    "default": DEFAULT_STATE_VECTOR_SPEC,
    "steer_history": STEER_HISTORY_STATE_VECTOR_SPEC,
}
STATE_VECTOR_SPEC = DEFAULT_STATE_VECTOR_SPEC
STATE_FEATURE_NAMES = STATE_VECTOR_SPEC.names
STATE_FEATURE_COUNT = STATE_VECTOR_SPEC.count
STATE_SPEED_NORMALIZER_KPH = STATE_VECTOR_SPEC.speed_normalizer_kph
DRIFT_DOUBLE_TAP_WINDOW_FRAMES = STATE_VECTOR_SPEC.shoulder_tap_guard_frames
RECENT_BOOST_PRESSURE_WINDOW_FRAMES = STATE_VECTOR_SPEC.recent_boost_window_frames
RECENT_STEER_PRESSURE_WINDOW_FRAMES = STATE_VECTOR_SPEC.recent_steer_window_frames
STATE_FEATURE_LOW = STATE_VECTOR_SPEC.low_array()
STATE_FEATURE_HIGH = STATE_VECTOR_SPEC.high_array()


def build_observation(
    *,
    image: np.ndarray,
    telemetry: FZeroXTelemetry | None,
    mode: ObservationMode,
    state_profile: ObservationStateProfile = DEFAULT_OBSERVATION_STATE_PROFILE,
    left_drift_held: float = 0.0,
    right_drift_held: float = 0.0,
    left_press_age_norm: float = 1.0,
    right_press_age_norm: float = 1.0,
    recent_boost_pressure: float = 0.0,
    steer_left_held: float = 0.0,
    steer_right_held: float = 0.0,
    recent_steer_pressure: float = 0.0,
) -> ObservationValue:
    if mode == "image":
        return image
    if mode == "image_state":
        return {
            "image": image,
            "state": telemetry_state_vector(
                telemetry,
                state_profile=state_profile,
                left_drift_held=left_drift_held,
                right_drift_held=right_drift_held,
                left_press_age_norm=left_press_age_norm,
                right_press_age_norm=right_press_age_norm,
                recent_boost_pressure=recent_boost_pressure,
                steer_left_held=steer_left_held,
                steer_right_held=steer_right_held,
                recent_steer_pressure=recent_steer_pressure,
            ),
        }
    raise ValueError(f"Unsupported observation mode: {mode!r}")


def telemetry_state_vector(
    telemetry: FZeroXTelemetry | None,
    *,
    state_profile: ObservationStateProfile = DEFAULT_OBSERVATION_STATE_PROFILE,
    left_drift_held: float = 0.0,
    right_drift_held: float = 0.0,
    left_press_age_norm: float = 1.0,
    right_press_age_norm: float = 1.0,
    recent_boost_pressure: float = 0.0,
    steer_left_held: float = 0.0,
    steer_right_held: float = 0.0,
    recent_steer_pressure: float = 0.0,
) -> np.ndarray:
    """Build the normalized scalar policy-state vector from live game telemetry."""

    spec = state_vector_spec(state_profile)
    left_held = _clamp(float(left_drift_held), 0.0, 1.0)
    right_held = _clamp(float(right_drift_held), 0.0, 1.0)
    left_age = _clamp(float(left_press_age_norm), 0.0, 1.0)
    right_age = _clamp(float(right_press_age_norm), 0.0, 1.0)
    boost_pressure = _clamp(float(recent_boost_pressure), 0.0, 1.0)
    steer_left = _clamp(float(steer_left_held), 0.0, 1.0)
    steer_right = _clamp(float(steer_right_held), 0.0, 1.0)
    steer_pressure = _clamp(float(recent_steer_pressure), -1.0, 1.0)
    if telemetry is None:
        values = [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            left_held,
            right_held,
            left_age,
            right_age,
            boost_pressure,
        ]
    else:
        player = telemetry.player
        max_energy = float(player.max_energy)
        energy_frac = 0.0 if max_energy <= 0.0 else float(player.energy) / max_energy
        boost_active = 1.0 if telemetry_boost_active(telemetry) else 0.0
        values = [
            _clamp(float(player.speed_kph) / spec.speed_normalizer_kph, 0.0, 2.0),
            _clamp(energy_frac, 0.0, 1.0),
            1.0 if player.reverse_timer > 0 else 0.0,
            1.0 if player.airborne else 0.0,
            1.0 if player.can_boost else 0.0,
            boost_active,
            left_held,
            right_held,
            left_age,
            right_age,
            boost_pressure,
        ]

    if state_profile == "steer_history":
        values.extend([steer_left, steer_right, steer_pressure])

    return np.array(values, dtype=np.float32)


def build_observation_space(
    observation_spec: ObservationSpec,
    *,
    frame_stack: int,
    mode: ObservationMode,
    state_profile: ObservationStateProfile = DEFAULT_OBSERVATION_STATE_PROFILE,
) -> spaces.Box | spaces.Dict:
    image_space = build_image_observation_space(observation_spec, frame_stack=frame_stack)
    if mode == "image":
        return image_space
    if mode == "image_state":
        spec = state_vector_spec(state_profile)
        return spaces.Dict(
            {
                "image": image_space,
                "state": spaces.Box(
                    low=spec.low_array(),
                    high=spec.high_array(),
                    shape=(spec.count,),
                    dtype=np.float32,
                ),
            }
        )
    raise ValueError(f"Unsupported observation mode: {mode!r}")


def build_image_observation_space(
    observation_spec: ObservationSpec,
    *,
    frame_stack: int,
) -> spaces.Box:
    return spaces.Box(
        low=0,
        high=255,
        shape=image_observation_shape(observation_spec, frame_stack=frame_stack),
        dtype=np.uint8,
    )


def image_observation_shape(
    observation_spec: ObservationSpec,
    *,
    frame_stack: int,
) -> tuple[int, int, int]:
    return (
        observation_spec.height,
        observation_spec.width,
        observation_spec.channels * frame_stack,
    )


def observation_image(observation: ObservationValue) -> np.ndarray:
    if isinstance(observation, dict):
        image = observation.get("image")
        if not isinstance(image, np.ndarray):
            raise ValueError("Dict observation is missing ndarray key 'image'")
        return image
    return observation


def observation_state(observation: ObservationValue) -> np.ndarray | None:
    if not isinstance(observation, dict):
        return None
    state = observation.get("state")
    if state is None:
        return None
    if not isinstance(state, np.ndarray):
        raise ValueError("Dict observation key 'state' must be an ndarray")
    return state


def state_vector_spec(state_profile: ObservationStateProfile) -> StateVectorSpec:
    """Return the scalar-state schema selected by config."""

    try:
        return STATE_VECTOR_SPECS[state_profile]
    except KeyError as exc:
        raise ValueError(f"Unsupported observation state profile: {state_profile!r}") from exc


def state_feature_names(state_profile: ObservationStateProfile) -> tuple[str, ...]:
    """Return ordered scalar-state feature names for one profile."""

    return state_vector_spec(state_profile).names


def state_feature_count(state_profile: ObservationStateProfile) -> int:
    """Return scalar-state width for one profile."""

    return state_vector_spec(state_profile).count


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))
