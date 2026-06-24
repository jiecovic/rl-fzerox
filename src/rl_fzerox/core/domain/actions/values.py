# src/rl_fzerox/core/domain/actions/values.py
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

type ActionMaskPreset = Literal["unrestricted"]
type ActionBranchName = Literal[
    "steer",
    "gas",
    "air_brake",
    "boost",
    "lean",
    "lean_left",
    "lean_right",
    "spin",
    "pitch",
]
type ActionBranchValue = Literal[
    "idle",
    "engaged",
    "left",
    "right",
    "down_full",
    "down",
    "neutral",
    "up",
    "up_full",
]
type ActionMaskValue = int | ActionBranchValue
type ActionMaskSpec = ActionMaskPreset | tuple[ActionMaskValue, ...]


@dataclass(frozen=True, slots=True)
class ActionBranchSpec:
    """Named mask values and fixed index width for one discrete action branch."""

    name: ActionBranchName
    named_values: tuple[ActionBranchValue, ...] = ()
    index_count: int | None = None


_ACTION_BRANCH_SPECS: tuple[ActionBranchSpec, ...] = (
    ActionBranchSpec(name="steer"),
    ActionBranchSpec(
        name="gas",
        named_values=("idle", "engaged"),
        index_count=2,
    ),
    ActionBranchSpec(
        name="air_brake",
        named_values=("idle", "engaged"),
        index_count=2,
    ),
    ActionBranchSpec(
        name="boost",
        named_values=("idle", "engaged"),
        index_count=2,
    ),
    ActionBranchSpec(
        name="lean",
        named_values=("idle", "left", "right"),
        index_count=3,
    ),
    ActionBranchSpec(
        name="lean_left",
        named_values=("idle", "engaged"),
        index_count=2,
    ),
    ActionBranchSpec(
        name="lean_right",
        named_values=("idle", "engaged"),
        index_count=2,
    ),
    ActionBranchSpec(name="spin", index_count=3),
    ActionBranchSpec(
        name="pitch",
        named_values=("down_full", "down", "neutral", "up", "up_full"),
        index_count=5,
    ),
)

ACTION_BRANCH_SPECS: Mapping[str, ActionBranchSpec] = MappingProxyType(
    {spec.name: spec for spec in _ACTION_BRANCH_SPECS}
)
DISCRETE_ACTION_BRANCH_VALUES: Mapping[str, tuple[ActionBranchValue, ...]] = MappingProxyType(
    {spec.name: spec.named_values for spec in _ACTION_BRANCH_SPECS if spec.named_values}
)


def compile_action_mask_values(branch_name: str, values: ActionMaskSpec) -> tuple[int, ...]:
    """Compile user-facing discrete action mask values to branch indices."""

    branch_spec = ACTION_BRANCH_SPECS.get(branch_name)
    if isinstance(values, str):
        return _compile_mask_preset(branch_name, values, branch_spec)

    indices: list[int] = []
    for mask_value in values:
        if isinstance(mask_value, bool):
            raise ValueError(f"Boolean mask value is not valid for action branch {branch_name!r}")
        if isinstance(mask_value, int):
            _validate_mask_index(mask_value, branch_name, branch_spec)
            indices.append(mask_value)
            continue
        if branch_spec is None or not branch_spec.named_values:
            raise ValueError(
                f"Named mask values are not supported for action branch {branch_name!r}"
            )
        if mask_value not in branch_spec.named_values:
            raise ValueError(f"Unknown mask value {mask_value!r} for action branch {branch_name!r}")
        indices.append(branch_spec.named_values.index(mask_value))
    if len(set(indices)) != len(indices):
        raise ValueError(f"Action mask for branch {branch_name!r} contains duplicates")
    return tuple(indices)


def _compile_mask_preset(
    branch_name: str,
    preset: ActionMaskPreset,
    branch_spec: ActionBranchSpec | None,
) -> tuple[int, ...]:
    if preset != "unrestricted":
        raise ValueError(f"Unknown mask preset {preset!r} for action branch {branch_name!r}")
    if branch_spec is None or branch_spec.index_count is None:
        raise ValueError(
            f"Mask preset {preset!r} is not supported for action branch {branch_name!r}"
        )
    return tuple(range(branch_spec.index_count))


def _validate_mask_index(
    mask_value: int,
    branch_name: str,
    branch_spec: ActionBranchSpec | None,
) -> None:
    if mask_value < 0:
        raise ValueError(
            f"Mask index {mask_value!r} is out of range for action branch {branch_name!r}"
        )
    if branch_spec is not None and branch_spec.index_count is not None:
        if mask_value >= branch_spec.index_count:
            raise ValueError(
                f"Mask index {mask_value!r} is out of range for action branch {branch_name!r}"
            )
