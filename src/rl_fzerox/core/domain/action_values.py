# src/rl_fzerox/core/domain/action_values.py
from __future__ import annotations

from typing import Literal, TypeAlias

ActionBranchValue: TypeAlias = Literal[
    "idle",
    "engaged",
    "left",
    "right",
]
ActionMaskValue: TypeAlias = int | ActionBranchValue

DISCRETE_ACTION_BRANCH_VALUES: dict[str, tuple[ActionBranchValue, ...]] = {
    "gas": ("idle", "engaged"),
    "air_brake": ("idle", "engaged"),
    "boost": ("idle", "engaged"),
    "lean": ("idle", "left", "right"),
}


def compile_action_mask_values(
    branch_name: str,
    values: tuple[ActionMaskValue, ...],
) -> tuple[int, ...]:
    """Compile user-facing discrete action mask values to branch indices."""

    expected_values = DISCRETE_ACTION_BRANCH_VALUES.get(branch_name)
    indices: list[int] = []
    for mask_value in values:
        if isinstance(mask_value, bool):
            raise ValueError(f"Boolean mask value is not valid for action branch {branch_name!r}")
        if isinstance(mask_value, int):
            if mask_value < 0:
                raise ValueError(
                    f"Mask index {mask_value!r} is out of range for action branch "
                    f"{branch_name!r}"
                )
            if expected_values is not None and mask_value >= len(expected_values):
                raise ValueError(
                    f"Mask index {mask_value!r} is out of range for action branch "
                    f"{branch_name!r}"
                )
            indices.append(mask_value)
            continue
        if expected_values is None:
            raise ValueError(
                f"Named mask values are not supported for action branch {branch_name!r}"
            )
        if mask_value not in expected_values:
            raise ValueError(
                f"Unknown mask value {mask_value!r} for action branch {branch_name!r}"
            )
        indices.append(expected_values.index(mask_value))
    if len(set(indices)) != len(indices):
        raise ValueError(f"Action mask for branch {branch_name!r} contains duplicates")
    return tuple(indices)
