# src/rl_fzerox/ui/watch/view/panels/content/state_vector_panel/model.py
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fzerox_emulator.arrays import StateVector
from rl_fzerox.core.policy.auxiliary_state.targets import AuxiliaryStateTargetName
from rl_fzerox.core.runtime_spec.schema import PolicyConfig


@dataclass(frozen=True, slots=True)
class AuxiliaryLossSpec:
    name: AuxiliaryStateTargetName
    grounded_only: bool


@dataclass(frozen=True, slots=True)
class StateVectorColumnLayout:
    pred_width: int = 6
    ref_width: int = 6
    err_width: int = 5
    obs_width: int = 6


@dataclass(frozen=True, slots=True)
class StateVectorGroup:
    title: str
    prefix: str | None
    component_name: str | None


@dataclass(frozen=True, slots=True)
class StateFeatureRow:
    index: int
    name: str
    observation_value: float | None
    reference_value: float | None


STATE_VECTOR_COLUMNS = StateVectorColumnLayout()
COMPONENT_GROUPS: tuple[StateVectorGroup, ...] = (
    StateVectorGroup("Vehicle", "vehicle_state.", "vehicle_state"),
    StateVectorGroup("Machine", "machine_context.", "machine_context"),
    StateVectorGroup("Track Position", "track_position.", "track_position"),
    StateVectorGroup("Surface", "surface_state.", "surface_state"),
    StateVectorGroup("Course", "course_context.", "course_context"),
)
CONTROL_HISTORY_GROUP = StateVectorGroup(
    "Control History",
    "control_history.",
    "control_history",
)
LEGACY_STATE_GROUP = StateVectorGroup("State", None, None)


def flattened_state_vector(observation_state: StateVector | None) -> StateVector | None:
    if observation_state is None:
        return None
    return np.asarray(observation_state, dtype=np.float32).reshape(-1)


def resolved_state_feature_names(
    feature_names: tuple[str, ...],
    observation_values: StateVector | None,
) -> tuple[str, ...]:
    if observation_values is None:
        return feature_names
    return (
        feature_names
        if len(feature_names) == observation_values.size
        else tuple(f"state_{index}" for index in range(observation_values.size))
    )


def auxiliary_loss_specs(policy_config: PolicyConfig | None) -> dict[str, AuxiliaryLossSpec]:
    if policy_config is None:
        return {}
    return {
        loss.name: AuxiliaryLossSpec(
            name=loss.name,
            grounded_only=bool(loss.grounded_only),
        )
        for loss in policy_config.auxiliary_state.losses
    }


def state_vector_groups(
    names: tuple[str, ...],
    *,
    auxiliary_loss_names: tuple[str, ...] = (),
) -> tuple[StateVectorGroup, ...]:
    used_component_names = {
        name
        for group in COMPONENT_GROUPS
        for name in names
        if group.prefix is not None and name.startswith(group.prefix)
    }
    groups: tuple[StateVectorGroup, ...] = tuple(
        group
        for group in COMPONENT_GROUPS
        if group.prefix is not None
        and (
            any(name.startswith(group.prefix) for name in names)
            or any(
                auxiliary_name_matches_group(name, group.prefix) for name in auxiliary_loss_names
            )
        )
    )
    if any(is_control_history_feature(name) for name in names):
        groups = (*groups, CONTROL_HISTORY_GROUP)
    legacy_names = tuple(
        name
        for name in names
        if name not in used_component_names and not is_control_history_feature(name)
    )
    if legacy_names:
        return (*groups, LEGACY_STATE_GROUP)
    return groups


def state_feature_rows(
    *,
    names: tuple[str, ...],
    observation_values: StateVector | None,
    reference_values: StateVector | None,
    group_prefix: str | None,
) -> tuple[StateFeatureRow, ...]:
    if observation_values is None:
        return ()

    rows: list[StateFeatureRow] = []
    for index, name in enumerate(names):
        if not state_vector_name_matches_group(name, group_prefix):
            continue
        observation_value = (
            None if index >= int(observation_values.size) else float(observation_values[index])
        )
        reference_value = (
            observation_value
            if reference_values is None or index >= int(reference_values.size)
            else float(reference_values[index])
        )
        rows.append(
            StateFeatureRow(
                index=index,
                name=name,
                observation_value=observation_value,
                reference_value=reference_value,
            )
        )
    return tuple(rows)


def state_vector_name_matches_group(name: str, group_prefix: str | None) -> bool:
    if group_prefix is None:
        return "." not in name and not name.startswith("prev_")
    return name.startswith(group_prefix)


def auxiliary_name_matches_group(name: str, group_prefix: str | None) -> bool:
    if group_prefix is None:
        return "." not in name
    return name.startswith(group_prefix)


def state_vector_label(name: str, *, group_prefix: str | None) -> str:
    if group_prefix is None:
        return name
    if group_prefix == "control_history." and name.startswith("prev_"):
        return name
    return name.removeprefix(group_prefix)


def is_control_history_feature(name: str) -> bool:
    return name.startswith("control_history.") or name.startswith("prev_")
