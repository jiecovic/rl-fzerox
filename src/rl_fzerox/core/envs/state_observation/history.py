# src/rl_fzerox/core/envs/state_observation/history.py
from __future__ import annotations

from collections.abc import Mapping

from rl_fzerox.core.envs.state_observation.types import ActionHistoryControl, StateFeature
from rl_fzerox.core.envs.state_observation.utils import clamp

ACTION_HISTORY_FEATURE_BOUNDS: dict[ActionHistoryControl, StateFeature] = {
    "steer": StateFeature("steer", 1.0, low=-1.0),
    "gas": StateFeature("gas", 1.0),
    "thrust": StateFeature("thrust", 1.0),
    "air_brake": StateFeature("air_brake", 1.0),
    "boost": StateFeature("boost", 1.0),
    "lean": StateFeature("lean", 1.0, low=-1.0),
    "pitch": StateFeature("pitch", 1.0, low=-1.0),
}


def action_history_feature_names(
    action_history_len: int | None,
    *,
    action_history_controls: tuple[ActionHistoryControl, ...],
) -> tuple[str, ...]:
    """Return ordered feature names for the configured previous-action buffer."""

    if action_history_len is None:
        return ()
    return tuple(
        feature.name
        for feature in action_history_features(
            action_history_len,
            action_history_controls=action_history_controls,
        )
    )


def action_history_features(
    action_history_len: int,
    *,
    action_history_controls: tuple[ActionHistoryControl, ...],
) -> tuple[StateFeature, ...]:
    length = validate_action_history_len(action_history_len)
    controls = validate_action_history_controls(action_history_controls)
    return tuple(
        StateFeature(f"prev_{base_feature.name}_{age}", base_feature.high, low=base_feature.low)
        for base_feature in (ACTION_HISTORY_FEATURE_BOUNDS[control] for control in controls)
        for age in range(1, length + 1)
    )


def action_history_values(
    action_history: Mapping[str, float],
    *,
    action_history_len: int,
    action_history_controls: tuple[ActionHistoryControl, ...],
) -> list[float]:
    values: list[float] = []
    for feature in action_history_features(
        action_history_len,
        action_history_controls=action_history_controls,
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
) -> tuple[StateFeature, ...]:
    length = validate_action_history_len(action_history_len)
    return tuple(
        StateFeature(
            f"control_history.prev_{control_history_feature_name(control)}_{age}",
            control_history_feature_bound(control).high,
            low=control_history_feature_bound(control).low,
        )
        for control in controls
        for age in range(1, length + 1)
    )


def component_action_history_values(
    action_history: Mapping[str, float],
    *,
    action_history_len: int,
    controls: tuple[ActionHistoryControl, ...],
) -> list[float]:
    values: list[float] = []
    for control in controls:
        bounds = control_history_feature_bound(control)
        source_control = control_history_source_control(control)
        source_name = ACTION_HISTORY_FEATURE_BOUNDS[source_control].name
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
