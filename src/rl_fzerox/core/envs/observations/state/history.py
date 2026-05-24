# src/rl_fzerox/core/envs/observations/state/history.py
from __future__ import annotations

from collections.abc import Mapping

from .types import ActionHistoryControl, StateFeature
from .utils import clamp

ACTION_HISTORY_FEATURE_BOUNDS: dict[ActionHistoryControl, StateFeature] = {
    "steer": StateFeature("steer", 1.0, low=-1.0),
    "gas": StateFeature("gas", 1.0),
    "thrust": StateFeature("thrust", 1.0),
    "air_brake": StateFeature("air_brake", 1.0),
    "boost": StateFeature("boost", 1.0),
    "pitch": StateFeature("pitch", 1.0, low=-1.0),
}


def action_history_feature_names(
    action_history_len: int | None,
    *,
    action_history_controls: tuple[ActionHistoryControl, ...],
    split_lean_history: bool = False,
) -> tuple[str, ...]:
    """Return ordered feature names for the configured previous-action buffer."""

    if action_history_len is None:
        return ()
    return tuple(
        feature.name
        for feature in action_history_features(
            action_history_len,
            action_history_controls=action_history_controls,
            split_lean_history=split_lean_history,
        )
    )


def action_history_features(
    action_history_len: int,
    *,
    action_history_controls: tuple[ActionHistoryControl, ...],
    split_lean_history: bool = False,
) -> tuple[StateFeature, ...]:
    length = validate_action_history_len(action_history_len)
    controls = validate_action_history_controls(action_history_controls)
    return tuple(
        StateFeature(f"prev_{base_feature.name}_{age}", base_feature.high, low=base_feature.low)
        for base_feature in _history_feature_bounds_for_controls(
            controls,
            split_lean_history=split_lean_history,
        )
        for age in range(1, length + 1)
    )


def action_history_values(
    action_history: Mapping[str, float],
    *,
    action_history_len: int,
    action_history_controls: tuple[ActionHistoryControl, ...],
    split_lean_history: bool = False,
) -> list[float]:
    values: list[float] = []
    for feature in action_history_features(
        action_history_len,
        action_history_controls=action_history_controls,
        split_lean_history=split_lean_history,
    ):
        values.append(
            clamp(
                float(action_history.get(feature.name, 0.0)),
                feature.low,
                feature.high,
            )
        )
    return values


def component_action_history_features(
    action_history_len: int,
    *,
    controls: tuple[ActionHistoryControl, ...],
    split_lean_history: bool = False,
) -> tuple[StateFeature, ...]:
    length = validate_action_history_len(action_history_len)
    return tuple(
        StateFeature(
            f"{bounds.name}_{age}",
            bounds.high,
            low=bounds.low,
        )
        for control in controls
        for _, bounds in _component_history_feature_specs(
            control,
            split_lean_history=split_lean_history,
        )
        for age in range(1, length + 1)
    )


def component_action_history_values(
    action_history: Mapping[str, float],
    *,
    action_history_len: int,
    controls: tuple[ActionHistoryControl, ...],
    split_lean_history: bool = False,
) -> list[float]:
    values: list[float] = []
    for control in controls:
        feature_specs = _component_history_feature_specs(
            control,
            split_lean_history=split_lean_history,
        )
        for source_name, bounds in feature_specs:
            for age in range(1, validate_action_history_len(action_history_len) + 1):
                values.append(
                    clamp(
                        float(action_history.get(f"prev_{source_name}_{age}", 0.0)),
                        bounds.low,
                        bounds.high,
                    )
                )
    return values


def control_history_feature_name(control: ActionHistoryControl) -> str:
    return "thrust" if control == "gas" else control


def control_history_source_control(control: ActionHistoryControl) -> ActionHistoryControl:
    return "gas" if control == "thrust" else control


def control_history_feature_bound(control: ActionHistoryControl) -> StateFeature:
    source_control = control_history_source_control(control)
    return ACTION_HISTORY_FEATURE_BOUNDS[source_control]


def _history_feature_bounds_for_controls(
    controls: tuple[ActionHistoryControl, ...],
    *,
    split_lean_history: bool,
) -> tuple[StateFeature, ...]:
    features: list[StateFeature] = []
    for control in controls:
        if control == "lean":
            features.extend(_lean_history_feature_bounds(split_lean_history=split_lean_history))
            continue
        features.append(ACTION_HISTORY_FEATURE_BOUNDS[control])
    return tuple(features)


def _component_history_feature_specs(
    control: ActionHistoryControl,
    *,
    split_lean_history: bool,
) -> tuple[tuple[str, StateFeature], ...]:
    if control == "lean":
        if split_lean_history:
            return (
                ("lean_left", StateFeature("control_history.prev_lean_left", 1.0)),
                ("lean_right", StateFeature("control_history.prev_lean_right", 1.0)),
            )
        bounds = StateFeature("control_history.prev_lean", 1.0, low=-1.0)
        return (("lean", bounds),)
    bounds = control_history_feature_bound(control)
    return (
        (
            ACTION_HISTORY_FEATURE_BOUNDS[control_history_source_control(control)].name,
            StateFeature(
                f"control_history.prev_{control_history_feature_name(control)}",
                bounds.high,
                low=bounds.low,
            ),
        ),
    )


def _lean_history_feature_bounds(*, split_lean_history: bool) -> tuple[StateFeature, ...]:
    if split_lean_history:
        return (
            StateFeature("lean_left", 1.0),
            StateFeature("lean_right", 1.0),
        )
    return (StateFeature("lean", 1.0, low=-1.0),)


def action_history_control_name(value: object) -> ActionHistoryControl:
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
    if value == "pitch":
        return "pitch"
    raise ValueError(f"Unsupported action-history control: {value!r}")


def validate_action_history_len(action_history_len: int) -> int:
    length = int(action_history_len)
    if length <= 0:
        raise ValueError("action_history_len must be positive or None")
    return length


def validate_action_history_controls(
    action_history_controls: tuple[ActionHistoryControl, ...],
) -> tuple[ActionHistoryControl, ...]:
    if len(set(action_history_controls)) != len(action_history_controls):
        raise ValueError("action_history_controls must not contain duplicates")
    normalized = {"gas" if control == "thrust" else control for control in action_history_controls}
    if len(normalized) != len(action_history_controls):
        raise ValueError("action_history_controls cannot contain both gas and thrust")
    return action_history_controls
