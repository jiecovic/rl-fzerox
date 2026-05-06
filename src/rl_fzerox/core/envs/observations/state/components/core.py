# src/rl_fzerox/core/envs/observations/state/components/core.py
from __future__ import annotations

from collections.abc import Callable, Mapping
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
from rl_fzerox.core.envs.observations.state.types import (
    ActionHistoryControl,
    StateFeature,
    StateVectorSpec,
)


@dataclass(frozen=True, slots=True)
class StateComponentDefinition:
    """One scalar-state component with schema and value builders kept together."""

    features: Callable[..., tuple[StateFeature, ...]]
    values: Callable[..., list[float]]


def state_vector_spec_from_components(
    state_components: StateComponentsSettings,
    *,
    independent_lean_buttons: bool = False,
) -> StateVectorSpec:
    features: list[StateFeature] = []
    for component in state_components:
        features.extend(
            state_component_features(
                component,
                independent_lean_buttons=independent_lean_buttons,
            )
        )

    return StateVectorSpec(features=tuple(features))


def component_state_values(
    telemetry: FZeroXTelemetry | None,
    *,
    state_components: StateComponentsSettings,
    action_history: Mapping[str, float],
    independent_lean_buttons: bool = False,
) -> list[float]:
    values: list[float] = []
    for component in state_components:
        definition = state_component_definition(component)
        if component.name == "control_history":
            component_values = definition.values(
                telemetry,
                component,
                action_history,
                independent_lean_buttons=independent_lean_buttons,
            )
        else:
            component_values = definition.values(
                telemetry,
                component,
                action_history,
            )
        values.extend(component_values)

    return values


def action_history_settings_for_observation(
    *,
    state_components: StateComponentsSettings | None,
) -> tuple[int | None, tuple[ActionHistoryControl, ...]]:
    """Return the control-history buffer shape needed by the selected observation."""

    if state_components is None:
        return None, ()
    control_config = component_by_name(state_components, "control_history")
    if control_config is None:
        return None, ()
    length = component_int(control_config, "length", default=2)
    controls = component_controls(control_config, default=("steer", "thrust", "boost", "lean"))
    return length, tuple(control_history_source_control(control) for control in controls)


def state_component_features(
    component: ObservationStateComponentSettings,
    *,
    independent_lean_buttons: bool = False,
) -> tuple[StateFeature, ...]:
    definition = state_component_definition(component)
    if component.name != "control_history":
        return definition.features(component)
    return definition.features(
        component,
        independent_lean_buttons=independent_lean_buttons,
    )


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
