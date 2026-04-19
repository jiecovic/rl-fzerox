# src/rl_fzerox/core/envs/observation_state.py
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Literal, TypeAlias

import numpy as np

from fzerox_emulator import FZeroXTelemetry
from fzerox_emulator.arrays import Float32Array, StateVector
from rl_fzerox.core.domain.observation_components import (
    ActionHistoryControlName as ActionHistoryControl,
)
from rl_fzerox.core.domain.observation_components import (
    ObservationCourseContextName as ObservationCourseContext,
)
from rl_fzerox.core.domain.observation_components import (
    ObservationGroundEffectContextName as ObservationGroundEffectContext,
)
from rl_fzerox.core.domain.observation_components import (
    ObservationStateComponentSettings,
    StateComponentsSettings,
)
from rl_fzerox.core.domain.observation_components import (
    ObservationStateProfileName as ObservationStateProfile,
)
from rl_fzerox.core.envs.course_effects import (
    GROUND_EFFECT_FEATURES,
    CourseEffect,
    course_effect_raw,
    ground_effect_flags,
)
from rl_fzerox.core.envs.telemetry import telemetry_boost_active

ObservationMode: TypeAlias = Literal["image", "image_state"]
DEFAULT_OBSERVATION_STATE_PROFILE: ObservationStateProfile = "default"
DEFAULT_OBSERVATION_COURSE_CONTEXT: ObservationCourseContext = "none"
DEFAULT_OBSERVATION_GROUND_EFFECT_CONTEXT: ObservationGroundEffectContext = "none"
DEFAULT_ACTION_HISTORY_LEN: int | None = None
DEFAULT_ACTION_HISTORY_CONTROLS: tuple[ActionHistoryControl, ...] = (
    "steer",
    "gas",
    "boost",
    "lean",
)
BUILTIN_COURSE_COUNT = 24
DEFAULT_LATERAL_VELOCITY_NORMALIZER = 32.0
DEFAULT_SLIDING_LATERAL_VELOCITY_THRESHOLD = 8.0


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


@dataclass(frozen=True, slots=True)
class StateComponentDefinition:
    """One scalar-state component with schema and value builders kept together."""

    features: Callable[[ObservationStateComponentSettings], tuple[StateFeature, ...]]
    values: Callable[
        [
            FZeroXTelemetry | None,
            ObservationStateComponentSettings,
            Mapping[str, float],
            Mapping[str, float],
        ],
        list[float],
    ]


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
    "thrust": StateFeature("thrust", 1.0),
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


def telemetry_state_vector(
    telemetry: FZeroXTelemetry | None,
    *,
    state_profile: ObservationStateProfile = DEFAULT_OBSERVATION_STATE_PROFILE,
    course_context: ObservationCourseContext = DEFAULT_OBSERVATION_COURSE_CONTEXT,
    ground_effect_context: ObservationGroundEffectContext = (
        DEFAULT_OBSERVATION_GROUND_EFFECT_CONTEXT
    ),
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
    state_components: StateComponentsSettings | None = None,
) -> StateVector:
    """Build the normalized scalar policy-state vector from live game telemetry."""

    if state_components is not None:
        values = _component_state_values(
            telemetry,
            state_components=state_components,
            action_history=action_history or {},
            legacy_fields={
                "left_lean_held": left_lean_held,
                "right_lean_held": right_lean_held,
                "left_press_age_norm": left_press_age_norm,
                "right_press_age_norm": right_press_age_norm,
                "recent_boost_pressure": recent_boost_pressure,
                "steer_left_held": steer_left_held,
                "steer_right_held": steer_right_held,
                "recent_steer_pressure": recent_steer_pressure,
            },
        )
        expected_count = _state_vector_spec_from_components(state_components).count
        if len(values) != expected_count:
            raise ValueError(
                "Observation state component value count does not match feature spec: "
                f"{len(values)} != {expected_count}"
            )
        return np.array(values, dtype=np.float32)

    spec = state_vector_spec(
        state_profile,
        course_context=course_context,
        ground_effect_context=ground_effect_context,
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

    values.extend(_course_context_values(telemetry, course_context=course_context))
    values.extend(
        _ground_effect_context_values(
            telemetry,
            ground_effect_context=ground_effect_context,
        )
    )

    if action_history_len is not None:
        values.extend(
            _action_history_values(
                action_history or {},
                action_history_len=action_history_len,
                action_history_controls=action_history_controls,
            )
        )

    return np.array(values, dtype=np.float32)


def state_vector_spec(
    state_profile: ObservationStateProfile,
    *,
    course_context: ObservationCourseContext = DEFAULT_OBSERVATION_COURSE_CONTEXT,
    ground_effect_context: ObservationGroundEffectContext = (
        DEFAULT_OBSERVATION_GROUND_EFFECT_CONTEXT
    ),
    action_history_len: int | None = DEFAULT_ACTION_HISTORY_LEN,
    action_history_controls: tuple[ActionHistoryControl, ...] = DEFAULT_ACTION_HISTORY_CONTROLS,
    state_components: StateComponentsSettings | None = None,
) -> StateVectorSpec:
    """Return the scalar-state schema selected by config."""

    if state_components is not None:
        return _state_vector_spec_from_components(state_components)

    try:
        base_spec = STATE_VECTOR_SPECS[state_profile]
    except KeyError as exc:
        raise ValueError(f"Unsupported observation state profile: {state_profile!r}") from exc
    base_spec = _state_vector_spec_with_course_context(base_spec, course_context=course_context)
    base_spec = _state_vector_spec_with_ground_effect_context(
        base_spec,
        ground_effect_context=ground_effect_context,
    )
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
    course_context: ObservationCourseContext = DEFAULT_OBSERVATION_COURSE_CONTEXT,
    ground_effect_context: ObservationGroundEffectContext = (
        DEFAULT_OBSERVATION_GROUND_EFFECT_CONTEXT
    ),
    action_history_len: int | None = DEFAULT_ACTION_HISTORY_LEN,
    action_history_controls: tuple[ActionHistoryControl, ...] = DEFAULT_ACTION_HISTORY_CONTROLS,
    state_components: StateComponentsSettings | None = None,
) -> tuple[str, ...]:
    """Return ordered scalar-state feature names for one profile."""

    return state_vector_spec(
        state_profile,
        course_context=course_context,
        ground_effect_context=ground_effect_context,
        action_history_len=action_history_len,
        action_history_controls=action_history_controls,
        state_components=state_components,
    ).names


def state_feature_count(
    state_profile: ObservationStateProfile,
    *,
    course_context: ObservationCourseContext = DEFAULT_OBSERVATION_COURSE_CONTEXT,
    ground_effect_context: ObservationGroundEffectContext = (
        DEFAULT_OBSERVATION_GROUND_EFFECT_CONTEXT
    ),
    action_history_len: int | None = DEFAULT_ACTION_HISTORY_LEN,
    action_history_controls: tuple[ActionHistoryControl, ...] = DEFAULT_ACTION_HISTORY_CONTROLS,
    state_components: StateComponentsSettings | None = None,
) -> int:
    """Return scalar-state width for one profile."""

    return state_vector_spec(
        state_profile,
        course_context=course_context,
        ground_effect_context=ground_effect_context,
        action_history_len=action_history_len,
        action_history_controls=action_history_controls,
        state_components=state_components,
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


def action_history_settings_for_observation(
    *,
    state_components: StateComponentsSettings | None,
    fallback_len: int | None,
    fallback_controls: tuple[ActionHistoryControl, ...],
) -> tuple[int | None, tuple[ActionHistoryControl, ...]]:
    """Return the control-history buffer shape needed by the selected observation."""

    if state_components is None:
        return fallback_len, fallback_controls
    control_config = _component_by_name(state_components, "control_history")
    if control_config is None:
        return None, ()
    length = _component_int(control_config, "length", default=2)
    controls = _component_controls(control_config, default=("steer", "thrust", "boost", "lean"))
    return length, tuple(_control_history_source_control(control) for control in controls)


def _state_vector_spec_from_components(
    state_components: StateComponentsSettings,
) -> StateVectorSpec:
    features: list[StateFeature] = []
    for component in state_components:
        features.extend(_state_component_definition(component).features(component))

    return StateVectorSpec(
        features=tuple(features),
        speed_normalizer_kph=DEFAULT_STATE_VECTOR_SPEC.speed_normalizer_kph,
        lean_tap_guard_frames=DEFAULT_STATE_VECTOR_SPEC.lean_tap_guard_frames,
        recent_boost_window_frames=DEFAULT_STATE_VECTOR_SPEC.recent_boost_window_frames,
        recent_steer_window_frames=DEFAULT_STATE_VECTOR_SPEC.recent_steer_window_frames,
    )


def _component_state_values(
    telemetry: FZeroXTelemetry | None,
    *,
    state_components: StateComponentsSettings,
    action_history: Mapping[str, float],
    legacy_fields: Mapping[str, float],
) -> list[float]:
    values: list[float] = []
    for component in state_components:
        values.extend(
            _state_component_definition(component).values(
                telemetry,
                component,
                action_history,
                legacy_fields,
            )
        )

    return values


def _state_component_definition(
    component: ObservationStateComponentSettings,
) -> StateComponentDefinition:
    component_name = _component_name(component)
    try:
        return STATE_COMPONENT_DEFINITIONS[component_name]
    except KeyError as exc:
        raise ValueError(f"Unsupported state component: {component_name!r}") from exc


def _vehicle_component_features(
    component: ObservationStateComponentSettings,
) -> tuple[StateFeature, ...]:
    del component
    return _vehicle_state_features()


def _vehicle_component_values(
    telemetry: FZeroXTelemetry | None,
    component: ObservationStateComponentSettings,
    action_history: Mapping[str, float],
    legacy_fields: Mapping[str, float],
) -> list[float]:
    del component, action_history, legacy_fields
    return _vehicle_state_values(telemetry)


def _track_position_component_features(
    component: ObservationStateComponentSettings,
) -> tuple[StateFeature, ...]:
    del component
    return _track_position_features()


def _track_position_component_values(
    telemetry: FZeroXTelemetry | None,
    component: ObservationStateComponentSettings,
    action_history: Mapping[str, float],
    legacy_fields: Mapping[str, float],
) -> list[float]:
    del component, action_history, legacy_fields
    return _track_position_values(telemetry)


def _surface_state_component_features(
    component: ObservationStateComponentSettings,
) -> tuple[StateFeature, ...]:
    del component
    return _surface_state_features()


def _surface_state_component_values(
    telemetry: FZeroXTelemetry | None,
    component: ObservationStateComponentSettings,
    action_history: Mapping[str, float],
    legacy_fields: Mapping[str, float],
) -> list[float]:
    del component, action_history, legacy_fields
    return _surface_state_values(telemetry)


def _course_context_component_features(
    component: ObservationStateComponentSettings,
) -> tuple[StateFeature, ...]:
    encoding = _component_str(component, "encoding", default="one_hot_builtin")
    return _course_component_features(encoding)


def _course_context_component_values(
    telemetry: FZeroXTelemetry | None,
    component: ObservationStateComponentSettings,
    action_history: Mapping[str, float],
    legacy_fields: Mapping[str, float],
) -> list[float]:
    del action_history, legacy_fields
    encoding = _component_str(component, "encoding", default="one_hot_builtin")
    return _course_component_values(telemetry, encoding=encoding)


def _legacy_state_component_features(
    component: ObservationStateComponentSettings,
) -> tuple[StateFeature, ...]:
    # V4 LEGACY SHIM: old runs can still request compact legacy state profiles.
    profile = _component_str(component, "state_profile", default="race_core")
    return STATE_VECTOR_SPECS[_state_profile_name(profile)].features


def _legacy_state_component_values(
    telemetry: FZeroXTelemetry | None,
    component: ObservationStateComponentSettings,
    action_history: Mapping[str, float],
    legacy_fields: Mapping[str, float],
) -> list[float]:
    del action_history
    # V4 LEGACY SHIM: old runs can still request compact legacy state profiles.
    profile = _state_profile_name(_component_str(component, "state_profile", default="race_core"))
    return _legacy_state_profile_values(
        telemetry,
        profile=profile,
        legacy_fields=legacy_fields,
    )


def _control_history_component_features(
    component: ObservationStateComponentSettings,
) -> tuple[StateFeature, ...]:
    length = _component_int(component, "length", default=2)
    controls = _component_controls(
        component,
        default=("steer", "thrust", "boost", "lean"),
    )
    return _component_action_history_features(length, controls=controls)


def _control_history_component_values(
    telemetry: FZeroXTelemetry | None,
    component: ObservationStateComponentSettings,
    action_history: Mapping[str, float],
    legacy_fields: Mapping[str, float],
) -> list[float]:
    del telemetry, legacy_fields
    length = _component_int(component, "length", default=2)
    controls = _component_controls(
        component,
        default=("steer", "thrust", "boost", "lean"),
    )
    return _component_action_history_values(
        action_history,
        action_history_len=length,
        controls=controls,
    )


STATE_COMPONENT_DEFINITIONS: dict[str, StateComponentDefinition] = {
    "vehicle_state": StateComponentDefinition(
        features=_vehicle_component_features,
        values=_vehicle_component_values,
    ),
    "track_position": StateComponentDefinition(
        features=_track_position_component_features,
        values=_track_position_component_values,
    ),
    "surface_state": StateComponentDefinition(
        features=_surface_state_component_features,
        values=_surface_state_component_values,
    ),
    "course_context": StateComponentDefinition(
        features=_course_context_component_features,
        values=_course_context_component_values,
    ),
    "legacy_state": StateComponentDefinition(
        features=_legacy_state_component_features,
        values=_legacy_state_component_values,
    ),
    "control_history": StateComponentDefinition(
        features=_control_history_component_features,
        values=_control_history_component_values,
    ),
}


def _vehicle_state_features() -> tuple[StateFeature, ...]:
    return (
        StateFeature("vehicle_state.speed_norm", 2.0),
        StateFeature("vehicle_state.energy_frac", 1.0),
        StateFeature("vehicle_state.reverse_active", 1.0),
        StateFeature("vehicle_state.airborne", 1.0),
        StateFeature("vehicle_state.boost_ready", 1.0),
        StateFeature("vehicle_state.boost_active", 1.0),
        StateFeature("vehicle_state.lateral_velocity_norm", 1.0, low=-1.0),
        StateFeature("vehicle_state.sliding_active", 1.0),
    )


def _vehicle_state_values(
    telemetry: FZeroXTelemetry | None,
) -> list[float]:
    if telemetry is None:
        return [0.0] * len(_vehicle_state_features())
    player = telemetry.player
    energy_frac = 0.0 if player.max_energy <= 0.0 else player.energy / player.max_energy
    lateral_velocity = float(player.local_lateral_velocity)
    return [
        _clamp(float(player.speed_kph) / STATE_SPEED_NORMALIZER_KPH, 0.0, 2.0),
        _clamp(float(energy_frac), 0.0, 1.0),
        1.0 if player.reverse_timer > 0 else 0.0,
        1.0 if player.airborne else 0.0,
        1.0 if player.can_boost else 0.0,
        1.0 if telemetry_boost_active(telemetry) else 0.0,
        _clamp(lateral_velocity / DEFAULT_LATERAL_VELOCITY_NORMALIZER, -1.0, 1.0),
        1.0
        if (
            not player.airborne
            and abs(lateral_velocity) > DEFAULT_SLIDING_LATERAL_VELOCITY_THRESHOLD
        )
        else 0.0,
    ]


def _track_position_features() -> tuple[StateFeature, ...]:
    return (
        StateFeature("track_position.edge_ratio", 1.0, low=-1.0),
        StateFeature("track_position.outside_track_bounds", 1.0),
    )


def _track_position_values(telemetry: FZeroXTelemetry | None) -> list[float]:
    edge_ratio = _raw_edge_ratio(telemetry)
    if edge_ratio is None:
        return [0.0, 0.0]
    return [
        _clamp(edge_ratio, -1.0, 1.0),
        1.0 if abs(edge_ratio) > 1.0 else 0.0,
    ]


def _raw_edge_ratio(telemetry: FZeroXTelemetry | None) -> float | None:
    if telemetry is None:
        return None
    player = telemetry.player
    offset = float(player.signed_lateral_offset)
    radius = (
        float(player.current_radius_left) if offset >= 0.0 else float(player.current_radius_right)
    )
    if radius <= 0.0:
        return None
    return offset / radius


def _surface_state_features() -> tuple[StateFeature, ...]:
    return (
        StateFeature("surface_state.on_refill_surface", 1.0),
        StateFeature("surface_state.on_dirt_surface", 1.0),
        StateFeature("surface_state.on_ice_surface", 1.0),
    )


def _surface_state_values(telemetry: FZeroXTelemetry | None) -> list[float]:
    raw_effect = course_effect_raw(telemetry)
    return [
        1.0 if telemetry is not None and telemetry.player.on_energy_refill else 0.0,
        1.0 if raw_effect == CourseEffect.DIRT else 0.0,
        1.0 if raw_effect == CourseEffect.ICE else 0.0,
    ]


def _course_component_features(encoding: str) -> tuple[StateFeature, ...]:
    if encoding == "none":
        return ()
    if encoding == "one_hot_builtin":
        return tuple(
            StateFeature(f"course_context.course_builtin_{index:02d}", 1.0)
            for index in range(BUILTIN_COURSE_COUNT)
        )
    raise ValueError(f"Unsupported course-context encoding: {encoding!r}")


def _course_component_values(
    telemetry: FZeroXTelemetry | None,
    *,
    encoding: str,
) -> list[float]:
    if encoding == "none":
        return []
    if encoding != "one_hot_builtin":
        raise ValueError(f"Unsupported course-context encoding: {encoding!r}")
    values = [0.0] * BUILTIN_COURSE_COUNT
    if telemetry is None:
        return values
    course_index = int(telemetry.course_index)
    if 0 <= course_index < BUILTIN_COURSE_COUNT:
        values[course_index] = 1.0
    return values


def _legacy_state_profile_values(
    telemetry: FZeroXTelemetry | None,
    *,
    profile: ObservationStateProfile,
    legacy_fields: Mapping[str, float],
) -> list[float]:
    return list(
        telemetry_state_vector(
            telemetry,
            state_profile=profile,
            action_history_len=None,
            left_lean_held=float(legacy_fields.get("left_lean_held", 0.0)),
            right_lean_held=float(legacy_fields.get("right_lean_held", 0.0)),
            left_press_age_norm=float(legacy_fields.get("left_press_age_norm", 1.0)),
            right_press_age_norm=float(legacy_fields.get("right_press_age_norm", 1.0)),
            recent_boost_pressure=float(legacy_fields.get("recent_boost_pressure", 0.0)),
            steer_left_held=float(legacy_fields.get("steer_left_held", 0.0)),
            steer_right_held=float(legacy_fields.get("steer_right_held", 0.0)),
            recent_steer_pressure=float(legacy_fields.get("recent_steer_pressure", 0.0)),
        )
    )


def _component_action_history_features(
    action_history_len: int,
    *,
    controls: tuple[ActionHistoryControl, ...],
) -> tuple[StateFeature, ...]:
    length = _validate_action_history_len(action_history_len)
    return tuple(
        StateFeature(
            f"control_history.prev_{_control_history_feature_name(control)}_{age}",
            _control_history_feature_bound(control).high,
            low=_control_history_feature_bound(control).low,
        )
        for control in controls
        for age in range(1, length + 1)
    )


def _component_action_history_values(
    action_history: Mapping[str, float],
    *,
    action_history_len: int,
    controls: tuple[ActionHistoryControl, ...],
) -> list[float]:
    values: list[float] = []
    for control in controls:
        bounds = _control_history_feature_bound(control)
        source_control = _control_history_source_control(control)
        source_name = ACTION_HISTORY_FEATURE_BOUNDS[source_control].name
        for age in range(1, _validate_action_history_len(action_history_len) + 1):
            values.append(
                _clamp(
                    float(action_history.get(f"prev_{source_name}_{age}", 0.0)),
                    bounds.low,
                    bounds.high,
                )
            )
    return values


def _control_history_feature_name(control: ActionHistoryControl) -> str:
    return "thrust" if control == "gas" else control


def _control_history_source_control(control: ActionHistoryControl) -> ActionHistoryControl:
    return "gas" if control == "thrust" else control


def _control_history_feature_bound(control: ActionHistoryControl) -> StateFeature:
    source_control = _control_history_source_control(control)
    return ACTION_HISTORY_FEATURE_BOUNDS[source_control]


def _component_by_name(
    state_components: StateComponentsSettings,
    component_name: str,
) -> ObservationStateComponentSettings | None:
    for component in state_components:
        if _component_name(component) == component_name:
            return component
    return None


def _component_name(component: ObservationStateComponentSettings) -> str:
    return component.name


def _component_int(
    component: ObservationStateComponentSettings,
    key: str,
    *,
    default: int,
) -> int:
    value = getattr(component, key)
    if isinstance(value, bool):
        return default
    if isinstance(value, int | float):
        return int(value)
    return default


def _component_str(
    component: ObservationStateComponentSettings,
    key: str,
    *,
    default: str,
) -> str:
    value = getattr(component, key)
    return value if isinstance(value, str) else default


def _component_controls(
    component: ObservationStateComponentSettings,
    *,
    default: tuple[ActionHistoryControl, ...],
) -> tuple[ActionHistoryControl, ...]:
    value = component.controls
    if not isinstance(value, list | tuple):
        return default
    return _validate_action_history_controls(
        tuple(_action_history_control_name(item) for item in value)
    )


def _action_history_control_name(value: object) -> ActionHistoryControl:
    if value == "steer":
        return "steer"
    if value == "gas":
        return "gas"
    if value == "thrust":
        return "thrust"
    if value == "air_brake":
        return "air_brake"
    if value == "boost":
        return "boost"
    if value == "lean":
        return "lean"
    raise ValueError(f"Unsupported action-history control: {value!r}")


def _state_profile_name(value: str) -> ObservationStateProfile:
    if value == "default":
        return "default"
    if value == "steer_history":
        return "steer_history"
    if value == "race_core":
        return "race_core"
    raise ValueError(f"Unsupported observation state profile: {value!r}")


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


def _state_vector_spec_with_course_context(
    base_spec: StateVectorSpec,
    *,
    course_context: ObservationCourseContext,
) -> StateVectorSpec:
    return StateVectorSpec(
        features=(
            *base_spec.features,
            *_course_context_features(course_context),
        ),
        speed_normalizer_kph=base_spec.speed_normalizer_kph,
        lean_tap_guard_frames=base_spec.lean_tap_guard_frames,
        recent_boost_window_frames=base_spec.recent_boost_window_frames,
        recent_steer_window_frames=base_spec.recent_steer_window_frames,
    )


def _state_vector_spec_with_ground_effect_context(
    base_spec: StateVectorSpec,
    *,
    ground_effect_context: ObservationGroundEffectContext,
) -> StateVectorSpec:
    return StateVectorSpec(
        features=(
            *base_spec.features,
            *_ground_effect_context_features(ground_effect_context),
        ),
        speed_normalizer_kph=base_spec.speed_normalizer_kph,
        lean_tap_guard_frames=base_spec.lean_tap_guard_frames,
        recent_boost_window_frames=base_spec.recent_boost_window_frames,
        recent_steer_window_frames=base_spec.recent_steer_window_frames,
    )


def _course_context_features(
    course_context: ObservationCourseContext,
) -> tuple[StateFeature, ...]:
    if course_context == "none":
        return ()
    if course_context == "one_hot_builtin":
        return tuple(
            StateFeature(f"course_builtin_{index:02d}", 1.0)
            for index in range(BUILTIN_COURSE_COUNT)
        )
    raise ValueError(f"Unsupported observation course context: {course_context!r}")


def _course_context_values(
    telemetry: FZeroXTelemetry | None,
    *,
    course_context: ObservationCourseContext,
) -> list[float]:
    if course_context == "none":
        return []
    if course_context != "one_hot_builtin":
        raise ValueError(f"Unsupported observation course context: {course_context!r}")
    values = [0.0] * BUILTIN_COURSE_COUNT
    if telemetry is None:
        return values
    course_index = int(telemetry.course_index)
    if 0 <= course_index < BUILTIN_COURSE_COUNT:
        values[course_index] = 1.0
    return values


def _ground_effect_context_features(
    ground_effect_context: ObservationGroundEffectContext,
) -> tuple[StateFeature, ...]:
    if ground_effect_context == "none":
        return ()
    if ground_effect_context == "effect_flags":
        return tuple(StateFeature(name, 1.0) for name in GROUND_EFFECT_FEATURES)
    raise ValueError(f"Unsupported observation ground-effect context: {ground_effect_context!r}")


def _ground_effect_context_values(
    telemetry: FZeroXTelemetry | None,
    *,
    ground_effect_context: ObservationGroundEffectContext,
) -> list[float]:
    if ground_effect_context == "none":
        return []
    if ground_effect_context != "effect_flags":
        raise ValueError(
            f"Unsupported observation ground-effect context: {ground_effect_context!r}"
        )
    return list(ground_effect_flags(telemetry))


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
    normalized = {"gas" if control == "thrust" else control for control in action_history_controls}
    if len(normalized) != len(action_history_controls):
        raise ValueError("action_history_controls cannot contain both gas and thrust")
    return action_history_controls


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))
