# src/rl_fzerox/core/envs/observations/state/components.py
from __future__ import annotations

from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass

from fzerox_emulator import FZeroXTelemetry
from rl_fzerox.core.domain.observation_components import (
    ObservationStateComponentSettings,
    StateComponentsSettings,
)
from rl_fzerox.core.envs.course_effects import CourseEffect, course_effect_raw
from rl_fzerox.core.envs.telemetry import telemetry_boost_active

from .contexts import (
    course_component_features,
    course_component_values,
)
from .history import (
    action_history_control_name,
    component_action_history_features,
    component_action_history_values,
    control_history_source_control,
    validate_action_history_controls,
)
from .profiles import DEFAULT_STATE_VECTOR_SPEC
from .types import (
    OBSERVATION_STATE_DEFAULTS,
    ActionHistoryControl,
    StateFeature,
    StateVectorSpec,
)
from .utils import clamp


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


@dataclass(frozen=True, slots=True)
class MachineContextNormalization:
    """Fixed stock-machine bounds used to scale vehicle setup into [0, 1]."""

    stat_max: float = 4.0
    weight_min: float = 780.0
    weight_max: float = 2340.0

    @property
    def weight_range(self) -> float:
        return self.weight_max - self.weight_min


MACHINE_CONTEXT_NORMALIZATION = MachineContextNormalization()


def state_vector_spec_from_components(
    state_components: StateComponentsSettings,
) -> StateVectorSpec:
    features: list[StateFeature] = []
    for component in state_components:
        features.extend(state_component_definition(component).features(component))

    return StateVectorSpec(
        features=tuple(features),
        speed_normalizer_kph=DEFAULT_STATE_VECTOR_SPEC.speed_normalizer_kph,
        lean_tap_guard_frames=DEFAULT_STATE_VECTOR_SPEC.lean_tap_guard_frames,
        recent_boost_window_frames=DEFAULT_STATE_VECTOR_SPEC.recent_boost_window_frames,
        recent_steer_window_frames=DEFAULT_STATE_VECTOR_SPEC.recent_steer_window_frames,
    )


def component_state_values(
    telemetry: FZeroXTelemetry | None,
    *,
    state_components: StateComponentsSettings,
    zeroed_state_components: Collection[str] = (),
    action_history: Mapping[str, float],
    profile_fields: Mapping[str, float],
) -> list[float]:
    values: list[float] = []
    for component in state_components:
        definition = state_component_definition(component)
        if component.name in zeroed_state_components:
            values.extend([0.0] * len(definition.features(component)))
        else:
            values.extend(
                definition.values(
                    telemetry,
                    component,
                    action_history,
                    profile_fields,
                )
            )

    return values


def action_history_settings_for_observation(
    *,
    state_components: StateComponentsSettings | None,
    fallback_len: int | None,
    fallback_controls: tuple[ActionHistoryControl, ...],
) -> tuple[int | None, tuple[ActionHistoryControl, ...]]:
    """Return the control-history buffer shape needed by the selected observation."""

    if state_components is None:
        return fallback_len, fallback_controls
    control_config = component_by_name(state_components, "control_history")
    if control_config is None:
        return None, ()
    length = component_int(control_config, "length", default=2)
    controls = component_controls(control_config, default=("steer", "thrust", "boost", "lean"))
    return length, tuple(control_history_source_control(control) for control in controls)


def state_component_definition(
    component: ObservationStateComponentSettings,
) -> StateComponentDefinition:
    component_name = component.name
    try:
        return STATE_COMPONENT_DEFINITIONS[component_name]
    except KeyError as exc:
        raise ValueError(f"Unsupported state component: {component_name!r}") from exc


def component_by_name(
    state_components: StateComponentsSettings,
    component_name: str,
) -> ObservationStateComponentSettings | None:
    for component in state_components:
        if component.name == component_name:
            return component
    return None


def component_int(
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


def component_str(
    component: ObservationStateComponentSettings,
    key: str,
    *,
    default: str,
) -> str:
    value = getattr(component, key)
    return value if isinstance(value, str) else default


def component_controls(
    component: ObservationStateComponentSettings,
    *,
    default: tuple[ActionHistoryControl, ...],
) -> tuple[ActionHistoryControl, ...]:
    value = component.controls
    if not isinstance(value, list | tuple):
        return default
    return validate_action_history_controls(
        tuple(action_history_control_name(item) for item in value)
    )


def _vehicle_component_features(
    component: ObservationStateComponentSettings,
) -> tuple[StateFeature, ...]:
    del component
    return _vehicle_state_features()


def _vehicle_component_values(
    telemetry: FZeroXTelemetry | None,
    component: ObservationStateComponentSettings,
    action_history: Mapping[str, float],
    profile_fields: Mapping[str, float],
) -> list[float]:
    del component, action_history, profile_fields
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
    profile_fields: Mapping[str, float],
) -> list[float]:
    del component, action_history, profile_fields
    return _track_position_values(telemetry)


def _machine_context_component_features(
    component: ObservationStateComponentSettings,
) -> tuple[StateFeature, ...]:
    del component
    return _machine_context_features()


def _machine_context_component_values(
    telemetry: FZeroXTelemetry | None,
    component: ObservationStateComponentSettings,
    action_history: Mapping[str, float],
    profile_fields: Mapping[str, float],
) -> list[float]:
    del component, action_history, profile_fields
    return _machine_context_values(telemetry)


def _surface_state_component_features(
    component: ObservationStateComponentSettings,
) -> tuple[StateFeature, ...]:
    del component
    return _surface_state_features()


def _surface_state_component_values(
    telemetry: FZeroXTelemetry | None,
    component: ObservationStateComponentSettings,
    action_history: Mapping[str, float],
    profile_fields: Mapping[str, float],
) -> list[float]:
    del component, action_history, profile_fields
    return _surface_state_values(telemetry)


def _course_context_component_features(
    component: ObservationStateComponentSettings,
) -> tuple[StateFeature, ...]:
    encoding = component_str(component, "encoding", default="one_hot_builtin")
    return course_component_features(encoding)


def _course_context_component_values(
    telemetry: FZeroXTelemetry | None,
    component: ObservationStateComponentSettings,
    action_history: Mapping[str, float],
    profile_fields: Mapping[str, float],
) -> list[float]:
    del action_history, profile_fields
    encoding = component_str(component, "encoding", default="one_hot_builtin")
    return course_component_values(telemetry, encoding=encoding)


def _control_history_component_features(
    component: ObservationStateComponentSettings,
) -> tuple[StateFeature, ...]:
    length = component_int(component, "length", default=2)
    controls = component_controls(
        component,
        default=("steer", "thrust", "boost", "lean"),
    )
    return component_action_history_features(length, controls=controls)


def _control_history_component_values(
    telemetry: FZeroXTelemetry | None,
    component: ObservationStateComponentSettings,
    action_history: Mapping[str, float],
    profile_fields: Mapping[str, float],
) -> list[float]:
    del telemetry, profile_fields
    length = component_int(component, "length", default=2)
    controls = component_controls(
        component,
        default=("steer", "thrust", "boost", "lean"),
    )
    return component_action_history_values(
        action_history,
        action_history_len=length,
        controls=controls,
    )


STATE_COMPONENT_DEFINITIONS: dict[str, StateComponentDefinition] = {
    "vehicle_state": StateComponentDefinition(
        features=_vehicle_component_features,
        values=_vehicle_component_values,
    ),
    "machine_context": StateComponentDefinition(
        features=_machine_context_component_features,
        values=_machine_context_component_values,
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
        StateFeature("vehicle_state.boost_unlocked", 1.0),
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
        clamp(float(player.speed_kph) / DEFAULT_STATE_VECTOR_SPEC.speed_normalizer_kph, 0.0, 2.0),
        clamp(float(energy_frac), 0.0, 1.0),
        1.0 if player.reverse_timer > 0 else 0.0,
        1.0 if player.airborne else 0.0,
        1.0 if player.can_boost else 0.0,
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


def _track_position_features() -> tuple[StateFeature, ...]:
    return (
        StateFeature("track_position.lap_progress", 1.0),
        StateFeature("track_position.edge_ratio", 1.0, low=-1.0),
        StateFeature("track_position.outside_track_bounds", 1.0),
    )


def _track_position_values(telemetry: FZeroXTelemetry | None) -> list[float]:
    lap_progress = _lap_progress_fraction(telemetry)
    edge_ratio = _raw_edge_ratio(telemetry)
    if edge_ratio is None:
        return [lap_progress, 0.0, 0.0]
    return [
        lap_progress,
        clamp(edge_ratio, -1.0, 1.0),
        1.0 if abs(edge_ratio) > 1.0 else 0.0,
    ]


def _lap_progress_fraction(telemetry: FZeroXTelemetry | None) -> float:
    if telemetry is None or telemetry.course_length <= 0.0:
        return 0.0
    return clamp(
        float(telemetry.player.lap_distance) / float(telemetry.course_length),
        0.0,
        1.0,
    )


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


def _machine_context_features() -> tuple[StateFeature, ...]:
    return (
        StateFeature("machine_context.body_stat", 1.0),
        StateFeature("machine_context.boost_stat", 1.0),
        StateFeature("machine_context.grip_stat", 1.0),
        StateFeature("machine_context.weight", 1.0),
        StateFeature("machine_context.engine", 1.0),
    )


def _machine_context_values(telemetry: FZeroXTelemetry | None) -> list[float]:
    if telemetry is None:
        return [0.0] * len(_machine_context_features())
    player = telemetry.player
    return [
        _machine_stat_value(float(player.machine_body_stat)),
        _machine_stat_value(float(player.machine_boost_stat)),
        _machine_stat_value(float(player.machine_grip_stat)),
        _machine_weight_value(float(player.machine_weight)),
        clamp(float(player.engine_setting), 0.0, 1.0),
    ]


def _machine_stat_value(raw_value: float) -> float:
    return clamp(raw_value / MACHINE_CONTEXT_NORMALIZATION.stat_max, 0.0, 1.0)


def _machine_weight_value(raw_value: float) -> float:
    return clamp(
        (raw_value - MACHINE_CONTEXT_NORMALIZATION.weight_min)
        / MACHINE_CONTEXT_NORMALIZATION.weight_range,
        0.0,
        1.0,
    )


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
