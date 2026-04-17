# src/rl_fzerox/core/envs/observations.py
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, TypeAlias, TypedDict

import numpy as np
from gymnasium import spaces

from fzerox_emulator import FZeroXTelemetry, ObservationSpec
from fzerox_emulator.arrays import Float32Array, ObservationFrame, StateVector
from rl_fzerox.core.envs.telemetry import telemetry_boost_active

ObservationMode: TypeAlias = Literal["image", "image_state"]
ObservationStateProfile: TypeAlias = Literal[
    "default",
    "steer_history",
    "race_core",
]
ActionHistoryControl: TypeAlias = Literal["steer", "gas", "air_brake", "boost", "lean"]
ImageObservation: TypeAlias = ObservationFrame


class ImageStateObservation(TypedDict):
    image: ObservationFrame
    state: StateVector


ObservationValue: TypeAlias = ImageObservation | ImageStateObservation
DEFAULT_OBSERVATION_STATE_PROFILE: ObservationStateProfile = "default"
DEFAULT_ACTION_HISTORY_LEN: int | None = None
DEFAULT_ACTION_HISTORY_CONTROLS: tuple[ActionHistoryControl, ...] = (
    "steer",
    "gas",
    "boost",
    "lean",
)


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
    lean_tap_guard_frames: int
    recent_boost_window_frames: int
    recent_steer_window_frames: int

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(feature.name for feature in self.features)

    @property
    def count(self) -> int:
        return len(self.features)

    def high_array(self) -> Float32Array:
        return np.array([feature.high for feature in self.features], dtype=np.float32)

    def low_array(self) -> Float32Array:
        return np.array([feature.low for feature in self.features], dtype=np.float32)


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
ACTION_HISTORY_FEATURE_BOUNDS: dict[ActionHistoryControl, StateFeature] = {
    "steer": StateFeature("steer", 1.0, low=-1.0),
    "gas": StateFeature("gas", 1.0),
    "air_brake": StateFeature("air_brake", 1.0),
    "boost": StateFeature("boost", 1.0),
    "lean": StateFeature("lean", 1.0, low=-1.0),
}
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
STATE_VECTOR_SPEC = DEFAULT_STATE_VECTOR_SPEC
STATE_FEATURE_NAMES = STATE_VECTOR_SPEC.names
STATE_FEATURE_COUNT = STATE_VECTOR_SPEC.count
STATE_SPEED_NORMALIZER_KPH = STATE_VECTOR_SPEC.speed_normalizer_kph
LEAN_DOUBLE_TAP_WINDOW_FRAMES = STATE_VECTOR_SPEC.lean_tap_guard_frames
RECENT_BOOST_PRESSURE_WINDOW_FRAMES = STATE_VECTOR_SPEC.recent_boost_window_frames
RECENT_STEER_PRESSURE_WINDOW_FRAMES = STATE_VECTOR_SPEC.recent_steer_window_frames
STATE_FEATURE_LOW = STATE_VECTOR_SPEC.low_array()
STATE_FEATURE_HIGH = STATE_VECTOR_SPEC.high_array()


def build_observation(
    *,
    image: ObservationFrame,
    telemetry: FZeroXTelemetry | None,
    mode: ObservationMode,
    state_profile: ObservationStateProfile = DEFAULT_OBSERVATION_STATE_PROFILE,
    left_lean_held: float = 0.0,
    right_lean_held: float = 0.0,
    left_press_age_norm: float = 1.0,
    right_press_age_norm: float = 1.0,
    recent_boost_pressure: float = 0.0,
    steer_left_held: float = 0.0,
    steer_right_held: float = 0.0,
    recent_steer_pressure: float = 0.0,
    action_history_len: int | None = DEFAULT_ACTION_HISTORY_LEN,
    action_history_controls: tuple[ActionHistoryControl, ...] = DEFAULT_ACTION_HISTORY_CONTROLS,
    action_history: Mapping[str, float] | None = None,
) -> ObservationValue:
    if mode == "image":
        return image
    if mode == "image_state":
        return {
            "image": image,
            "state": telemetry_state_vector(
                telemetry,
                state_profile=state_profile,
                left_lean_held=left_lean_held,
                right_lean_held=right_lean_held,
                left_press_age_norm=left_press_age_norm,
                right_press_age_norm=right_press_age_norm,
                recent_boost_pressure=recent_boost_pressure,
                steer_left_held=steer_left_held,
                steer_right_held=steer_right_held,
                recent_steer_pressure=recent_steer_pressure,
                action_history_len=action_history_len,
                action_history_controls=action_history_controls,
                action_history=action_history,
            ),
        }
    raise ValueError(f"Unsupported observation mode: {mode!r}")


def telemetry_state_vector(
    telemetry: FZeroXTelemetry | None,
    *,
    state_profile: ObservationStateProfile = DEFAULT_OBSERVATION_STATE_PROFILE,
    left_lean_held: float = 0.0,
    right_lean_held: float = 0.0,
    left_press_age_norm: float = 1.0,
    right_press_age_norm: float = 1.0,
    recent_boost_pressure: float = 0.0,
    steer_left_held: float = 0.0,
    steer_right_held: float = 0.0,
    recent_steer_pressure: float = 0.0,
    action_history_len: int | None = DEFAULT_ACTION_HISTORY_LEN,
    action_history_controls: tuple[ActionHistoryControl, ...] = DEFAULT_ACTION_HISTORY_CONTROLS,
    action_history: Mapping[str, float] | None = None,
) -> StateVector:
    """Build the normalized scalar policy-state vector from live game telemetry."""

    spec = state_vector_spec(
        state_profile,
        action_history_len=action_history_len,
        action_history_controls=action_history_controls,
    )
    left_held = _clamp(float(left_lean_held), 0.0, 1.0)
    right_held = _clamp(float(right_lean_held), 0.0, 1.0)
    left_age = _clamp(float(left_press_age_norm), 0.0, 1.0)
    right_age = _clamp(float(right_press_age_norm), 0.0, 1.0)
    boost_pressure = _clamp(float(recent_boost_pressure), 0.0, 1.0)
    steer_left = _clamp(float(steer_left_held), 0.0, 1.0)
    steer_right = _clamp(float(steer_right_held), 0.0, 1.0)
    steer_pressure = _clamp(float(recent_steer_pressure), -1.0, 1.0)
    if telemetry is None:
        race_core_values = [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
        values = [
            race_core_values[0],
            race_core_values[1],
            race_core_values[2],
            race_core_values[3],
            race_core_values[4],
            race_core_values[5],
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
        race_core_values = [
            _clamp(float(player.speed_kph) / spec.speed_normalizer_kph, 0.0, 2.0),
            _clamp(energy_frac, 0.0, 1.0),
            1.0 if player.reverse_timer > 0 else 0.0,
            1.0 if player.airborne else 0.0,
            1.0 if player.can_boost else 0.0,
            boost_active,
        ]
        values = [
            race_core_values[0],
            race_core_values[1],
            race_core_values[2],
            race_core_values[3],
            race_core_values[4],
            race_core_values[5],
            left_held,
            right_held,
            left_age,
            right_age,
            boost_pressure,
        ]

    if state_profile == "race_core":
        values = race_core_values
    elif state_profile == "steer_history":
        values.extend([steer_left, steer_right, steer_pressure])

    if action_history_len is not None:
        values.extend(
            _action_history_values(
                action_history or {},
                action_history_len=action_history_len,
                action_history_controls=action_history_controls,
            )
        )

    return np.array(values, dtype=np.float32)


def build_observation_space(
    observation_spec: ObservationSpec,
    *,
    frame_stack: int,
    mode: ObservationMode,
    state_profile: ObservationStateProfile = DEFAULT_OBSERVATION_STATE_PROFILE,
    action_history_len: int | None = DEFAULT_ACTION_HISTORY_LEN,
    action_history_controls: tuple[ActionHistoryControl, ...] = DEFAULT_ACTION_HISTORY_CONTROLS,
) -> spaces.Box | spaces.Dict:
    image_space = build_image_observation_space(observation_spec, frame_stack=frame_stack)
    if mode == "image":
        return image_space
    if mode == "image_state":
        spec = state_vector_spec(
            state_profile,
            action_history_len=action_history_len,
            action_history_controls=action_history_controls,
        )
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


def observation_image(observation: ObservationValue) -> ObservationFrame:
    if isinstance(observation, dict):
        image = observation.get("image")
        if not isinstance(image, np.ndarray):
            raise ValueError("Dict observation is missing ndarray key 'image'")
        return image
    return observation


def observation_state(observation: ObservationValue) -> StateVector | None:
    if not isinstance(observation, dict):
        return None
    state = observation.get("state")
    if state is None:
        return None
    if not isinstance(state, np.ndarray):
        raise ValueError("Dict observation key 'state' must be an ndarray")
    return state


def state_vector_spec(
    state_profile: ObservationStateProfile,
    *,
    action_history_len: int | None = DEFAULT_ACTION_HISTORY_LEN,
    action_history_controls: tuple[ActionHistoryControl, ...] = DEFAULT_ACTION_HISTORY_CONTROLS,
) -> StateVectorSpec:
    """Return the scalar-state schema selected by config."""

    try:
        base_spec = STATE_VECTOR_SPECS[state_profile]
    except KeyError as exc:
        raise ValueError(f"Unsupported observation state profile: {state_profile!r}") from exc
    if action_history_len is None:
        return base_spec
    return _state_vector_spec_with_action_history(
        base_spec,
        action_history_len,
        action_history_controls=action_history_controls,
    )


def state_feature_names(
    state_profile: ObservationStateProfile,
    *,
    action_history_len: int | None = DEFAULT_ACTION_HISTORY_LEN,
    action_history_controls: tuple[ActionHistoryControl, ...] = DEFAULT_ACTION_HISTORY_CONTROLS,
) -> tuple[str, ...]:
    """Return ordered scalar-state feature names for one profile."""

    return state_vector_spec(
        state_profile,
        action_history_len=action_history_len,
        action_history_controls=action_history_controls,
    ).names


def state_feature_count(
    state_profile: ObservationStateProfile,
    *,
    action_history_len: int | None = DEFAULT_ACTION_HISTORY_LEN,
    action_history_controls: tuple[ActionHistoryControl, ...] = DEFAULT_ACTION_HISTORY_CONTROLS,
) -> int:
    """Return scalar-state width for one profile."""

    return state_vector_spec(
        state_profile,
        action_history_len=action_history_len,
        action_history_controls=action_history_controls,
    ).count


def action_history_feature_names(
    action_history_len: int | None,
    *,
    action_history_controls: tuple[ActionHistoryControl, ...] = DEFAULT_ACTION_HISTORY_CONTROLS,
) -> tuple[str, ...]:
    """Return ordered feature names for the configured previous-action buffer."""

    if action_history_len is None:
        return ()
    return tuple(
        feature.name
        for feature in _action_history_features(
            action_history_len,
            action_history_controls=action_history_controls,
        )
    )


def _state_vector_spec_with_action_history(
    base_spec: StateVectorSpec,
    action_history_len: int,
    *,
    action_history_controls: tuple[ActionHistoryControl, ...],
) -> StateVectorSpec:
    return StateVectorSpec(
        features=(
            *base_spec.features,
            *_action_history_features(
                action_history_len,
                action_history_controls=action_history_controls,
            ),
        ),
        speed_normalizer_kph=base_spec.speed_normalizer_kph,
        lean_tap_guard_frames=base_spec.lean_tap_guard_frames,
        recent_boost_window_frames=base_spec.recent_boost_window_frames,
        recent_steer_window_frames=base_spec.recent_steer_window_frames,
    )


def _action_history_features(
    action_history_len: int,
    *,
    action_history_controls: tuple[ActionHistoryControl, ...],
) -> tuple[StateFeature, ...]:
    length = _validate_action_history_len(action_history_len)
    controls = _validate_action_history_controls(action_history_controls)
    return tuple(
        StateFeature(f"prev_{base_feature.name}_{age}", base_feature.high, low=base_feature.low)
        for base_feature in (ACTION_HISTORY_FEATURE_BOUNDS[control] for control in controls)
        for age in range(1, length + 1)
    )


def _action_history_values(
    action_history: Mapping[str, float],
    *,
    action_history_len: int,
    action_history_controls: tuple[ActionHistoryControl, ...],
) -> list[float]:
    values: list[float] = []
    for feature in _action_history_features(
        action_history_len,
        action_history_controls=action_history_controls,
    ):
        values.append(
            _clamp(
                float(action_history.get(feature.name, 0.0)),
                feature.low,
                feature.high,
            )
        )
    return values


def _validate_action_history_len(action_history_len: int) -> int:
    length = int(action_history_len)
    if length <= 0:
        raise ValueError("action_history_len must be positive or None")
    return length


def _validate_action_history_controls(
    action_history_controls: tuple[ActionHistoryControl, ...],
) -> tuple[ActionHistoryControl, ...]:
    if len(set(action_history_controls)) != len(action_history_controls):
        raise ValueError("action_history_controls must not contain duplicates")
    return action_history_controls


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))
