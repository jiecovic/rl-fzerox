# src/rl_fzerox/core/envs/observations/state/components/core.py
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

from fzerox_emulator import FZeroXTelemetry
from rl_fzerox.core.domain.observations import (
    ObservationStateComponentSettings,
    StateComponentsSettings,
    default_excluded_state_feature_names,
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
    split_lean_history: bool = False,
) -> StateVectorSpec:
    features: list[StateFeature] = []
    for component in state_components:
        features.extend(
            state_component_features(
                component,
                split_lean_history=split_lean_history,
            )
        )

    return StateVectorSpec(features=tuple(features))


def component_state_values(
    telemetry: FZeroXTelemetry | None,
    *,
    state_components: StateComponentsSettings,
    action_history: Mapping[str, float],
    split_lean_history: bool = False,
) -> list[float]:
    values: list[float] = []
    for component in state_components:
        raw_features = raw_state_component_features(
            component,
            split_lean_history=split_lean_history,
        )
        definition = state_component_definition(component)
        if component.name == "control_history":
            raw_values = definition.values(
                telemetry,
                component,
                action_history,
                split_lean_history=split_lean_history,
            )
        else:
            raw_values = definition.values(
                telemetry,
                component,
                action_history,
            )
        values.extend(selected_state_component_values(component, raw_features, raw_values))

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
    split_lean_history: bool = False,
) -> tuple[StateFeature, ...]:
    return selected_state_component_features(
        component,
        raw_state_component_features(
            component,
            split_lean_history=split_lean_history,
        ),
    )


def raw_state_component_features(
    component: ObservationStateComponentSettings,
    *,
    split_lean_history: bool = False,
) -> tuple[StateFeature, ...]:
    """Return every feature one component can emit before inclusion filtering."""

    definition = state_component_definition(component)
    if component.name != "control_history":
        return definition.features(component)
    return definition.features(
        component,
        split_lean_history=split_lean_history,
    )


def selected_state_component_features(
    component: ObservationStateComponentSettings,
    raw_features: tuple[StateFeature, ...],
) -> tuple[StateFeature, ...]:
    """Apply the component's explicit/default feature inclusion policy."""

    selected_indexes = selected_state_component_feature_indexes(component, raw_features)
    return tuple(raw_features[index] for index in selected_indexes)


def selected_state_component_values(
    component: ObservationStateComponentSettings,
    raw_features: tuple[StateFeature, ...],
    raw_values: list[float],
) -> list[float]:
    """Apply the same feature inclusion policy to runtime values."""

    if len(raw_features) != len(raw_values):
        raise ValueError(
            f"{component.name} emitted {len(raw_values)} value(s) for "
            f"{len(raw_features)} feature(s)"
        )
    selected_indexes = selected_state_component_feature_indexes(component, raw_features)
    return [raw_values[index] for index in selected_indexes]


def selected_state_component_feature_indexes(
    component: ObservationStateComponentSettings,
    raw_features: tuple[StateFeature, ...],
) -> tuple[int, ...]:
    raw_names = tuple(feature.name for feature in raw_features)
    raw_name_set = set(raw_names)
    if component.included_features is None:
        default_excluded = set(default_excluded_state_feature_names(component.name))
        return tuple(
            index
            for index, feature_name in enumerate(raw_names)
            if feature_name not in default_excluded
        )

    included = tuple(component.included_features)
    if len(set(included)) != len(included):
        raise ValueError(f"{component.name}.included_features must not contain duplicates")
    unsupported = sorted(set(included) - raw_name_set)
    if unsupported:
        joined = ", ".join(unsupported)
        raise ValueError(f"{component.name} does not support included feature(s): {joined}")
    included_set = set(included)
    return tuple(
        index for index, feature_name in enumerate(raw_names) if feature_name in included_set
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
