# src/rl_fzerox/core/envs/observations/state/components/core.py
from __future__ import annotations

from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass

from fzerox_emulator import FZeroXTelemetry
from rl_fzerox.core.domain.observation_components import (
    ObservationStateComponentSettings,
    StateComponentsSettings,
)
from rl_fzerox.core.envs.observations.state.history import (
    action_history_control_name,
    control_history_source_control,
    validate_action_history_controls,
)
from rl_fzerox.core.envs.observations.state.profiles import DEFAULT_STATE_VECTOR_SPEC
from rl_fzerox.core.envs.observations.state.types import (
    ActionHistoryControl,
    StateFeature,
    StateVectorSpec,
)


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
    zeroed_state_features: Collection[str] = (),
    action_history: Mapping[str, float],
    profile_fields: Mapping[str, float],
) -> list[float]:
    values: list[float] = []
    for component in state_components:
        definition = state_component_definition(component)
        component_features = definition.features(component)
        if component.name in zeroed_state_components:
            values.extend([0.0] * len(component_features))
            continue

        component_values = definition.values(
            telemetry,
            component,
            action_history,
            profile_fields,
        )
        values.extend(
            0.0 if feature.name in zeroed_state_features else value
            for feature, value in zip(component_features, component_values, strict=True)
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
    from rl_fzerox.core.envs.observations.state.components.registry import (
        STATE_COMPONENT_DEFINITIONS,
    )

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
