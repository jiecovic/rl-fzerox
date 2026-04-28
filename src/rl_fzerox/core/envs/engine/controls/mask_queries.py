from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from fzerox_emulator.arrays import ActionMask
from rl_fzerox.core.domain.hybrid_action import HYBRID_DISCRETE_ACTION_KEY
from rl_fzerox.core.envs.actions import ActionValue
from rl_fzerox.core.envs.actions.base import DiscreteActionDimension

from .mask_config import ActionMaskBranches


def split_action_mask_by_branch(
    dimensions: tuple[DiscreteActionDimension, ...],
    mask: ActionMask,
) -> ActionMaskBranches:
    """Split one flattened MultiDiscrete mask into stable branch labels."""

    flat_mask = tuple(bool(value) for value in mask.reshape(-1).tolist())
    branches: ActionMaskBranches = {}
    cursor = 0
    for dimension in dimensions:
        next_cursor = cursor + dimension.size
        branches[dimension.label] = flat_mask[cursor:next_cursor]
        cursor = next_cursor
    if cursor != len(flat_mask):
        raise RuntimeError(
            "Action mask length does not match adapter dimensions: "
            f"mask={len(flat_mask)}, dimensions={cursor}"
        )
    return branches


def action_branch_value_allowed(
    branches: ActionMaskBranches | None,
    label: str,
    index: int,
    *,
    missing_allowed: bool,
) -> bool:
    """Return whether one discrete branch value is selectable.

    `missing_allowed` is explicit because a missing branch means different
    things at different call sites: continuous gas has no discrete branch but is
    still available, while optional cockpit buttons with no branch are not wired
    into the policy at all.
    """

    if branches is None:
        return True
    branch = branches.get(label)
    if branch is None:
        return missing_allowed
    if not 0 <= index < len(branch):
        return False
    return branch[index]


def action_branch_non_neutral_allowed(
    branches: ActionMaskBranches | None,
    label: str,
    *,
    neutral_index: int,
    missing_allowed: bool,
) -> bool:
    """Return whether any non-neutral value is selectable for one branch."""

    if branches is None:
        return True
    branch = branches.get(label)
    if branch is None:
        return missing_allowed
    return any(allowed for index, allowed in enumerate(branch) if index != neutral_index)


def selected_action_branches(
    branches: ActionMaskBranches,
    action: ActionValue | None,
) -> dict[str, int]:
    """Return selected discrete branch values in action-mask branch order."""

    if action is None or not branches:
        return {}
    discrete_values = _discrete_action_values(action)
    if discrete_values is None:
        return {}
    labels = tuple(branches)
    return {
        label: int(discrete_values[index])
        for index, label in enumerate(labels)
        if index < len(discrete_values)
    }


def action_mask_violations(
    branches: ActionMaskBranches,
    action: ActionValue | None,
) -> tuple[str, ...]:
    """Return branch/value pairs where an action selects a masked value."""

    selected = selected_action_branches(branches, action)
    violations: list[str] = []
    for label, value in selected.items():
        branch = branches[label]
        if not 0 <= value < len(branch) or not branch[value]:
            violations.append(f"{label}={value}")
    return tuple(violations)


def _discrete_action_values(action: ActionValue) -> tuple[int, ...] | None:
    source: object
    if isinstance(action, Mapping):
        source = action.get(HYBRID_DISCRETE_ACTION_KEY)
        if source is None:
            return None
    else:
        source = action
    values = np.asarray(source).reshape(-1)
    if values.size == 0:
        return None
    return tuple(int(value) for value in values)
