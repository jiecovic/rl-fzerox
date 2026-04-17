# src/rl_fzerox/core/config/action_branches.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, NonNegativeFloat, PositiveFloat, field_validator

from rl_fzerox.core.domain.action_adapters import (
    ACTION_ADAPTER_HYBRID_STEER_GAS_AIR_BRAKE_BOOST_LEAN,
    ACTION_ADAPTER_HYBRID_STEER_GAS_BOOST_LEAN,
)
from rl_fzerox.core.domain.lean import LeanMode

ActionBranchType: TypeAlias = Literal["continuous", "discrete"]
ActionBranchValue: TypeAlias = Literal[
    "idle",
    "engaged",
    "left",
    "right",
]

_STEER_GAS_BOOST_LEAN_BRANCHES = ("steer", "gas", "boost", "lean")
_STEER_GAS_AIR_BRAKE_BOOST_LEAN_BRANCHES = (
    "steer",
    "gas",
    "air_brake",
    "boost",
    "lean",
)
_DISCRETE_BRANCH_VALUES: dict[str, tuple[ActionBranchValue, ...]] = {
    "gas": ("idle", "engaged"),
    "air_brake": ("idle", "engaged"),
    "boost": ("idle", "engaged"),
    "lean": ("idle", "left", "right"),
}


class ActionBranchConfig(BaseModel):
    """One composable policy action branch declared in YAML."""

    model_config = ConfigDict(extra="forbid")

    type: ActionBranchType
    mask: tuple[ActionBranchValue, ...] | None = None
    response_power: PositiveFloat | None = None
    mode: LeanMode | None = None
    unmask_max_speed_kph: NonNegativeFloat | None = None
    unmask_min_speed_kph: NonNegativeFloat | None = None

    @field_validator("mask")
    @classmethod
    def _validate_unique_values(
        cls,
        value: tuple[ActionBranchValue, ...] | None,
    ) -> tuple[ActionBranchValue, ...] | None:
        if value is not None and len(set(value)) != len(value):
            raise ValueError("action branch mask must not contain duplicates")
        return value


class ActionBranchesConfig(BaseModel):
    """Composable action-space declaration compiled to the current adapters."""

    model_config = ConfigDict(extra="forbid")

    steer: ActionBranchConfig | None = None
    gas: ActionBranchConfig | None = None
    air_brake: ActionBranchConfig | None = None
    boost: ActionBranchConfig | None = None
    lean: ActionBranchConfig | None = None


@dataclass(frozen=True, slots=True)
class ActionBranchCompilation:
    """Legacy adapter fields derived from a composable branch declaration."""

    values: dict[str, object]


def compile_action_branches(raw_branches: object) -> ActionBranchCompilation:
    """Compile branch-style YAML into existing adapter config fields."""

    branches = ActionBranchesConfig.model_validate(raw_branches)
    branch_names = _configured_branch_names(branches)
    if branch_names == _STEER_GAS_BOOST_LEAN_BRANCHES:
        adapter_name = ACTION_ADAPTER_HYBRID_STEER_GAS_BOOST_LEAN
    elif branch_names == _STEER_GAS_AIR_BRAKE_BOOST_LEAN_BRANCHES:
        adapter_name = ACTION_ADAPTER_HYBRID_STEER_GAS_AIR_BRAKE_BOOST_LEAN
    else:
        names = ", ".join(branch_names) or "none"
        raise ValueError(f"Unsupported action branch combination: {names}")

    compiled: dict[str, object] = {"name": adapter_name}
    steer_branch = _required_branch(branches, "steer")
    _validate_continuous_branch("steer", steer_branch)
    if steer_branch.response_power is not None:
        compiled["steer_response_power"] = steer_branch.response_power

    mask = _compile_action_mask(branches)
    if mask:
        compiled["mask"] = mask

    boost_branch = branches.boost
    if boost_branch is not None:
        compiled["boost_unmask_max_speed_kph"] = boost_branch.unmask_max_speed_kph

    lean_branch = branches.lean
    if lean_branch is not None:
        compiled["lean_unmask_min_speed_kph"] = lean_branch.unmask_min_speed_kph
        if lean_branch.mode is not None:
            compiled["lean_mode"] = lean_branch.mode

    return ActionBranchCompilation(values=compiled)


def _configured_branch_names(branches: ActionBranchesConfig) -> tuple[str, ...]:
    return tuple(
        branch_name
        for branch_name in ("steer", "gas", "air_brake", "boost", "lean")
        if getattr(branches, branch_name) is not None
    )


def _required_branch(branches: ActionBranchesConfig, branch_name: str) -> ActionBranchConfig:
    branch = getattr(branches, branch_name)
    if branch is None:
        raise ValueError(f"Missing action branch: {branch_name}")
    if not isinstance(branch, ActionBranchConfig):
        raise TypeError(f"Invalid action branch: {branch_name}")
    return branch


def _validate_continuous_branch(branch_name: str, branch: ActionBranchConfig) -> None:
    if branch.type != "continuous":
        raise ValueError(f"action branch {branch_name!r} must be continuous")
    if branch.mask is not None:
        raise ValueError(f"continuous action branch {branch_name!r} cannot define a mask")
    _validate_branch_specific_fields(branch_name, branch)


def _compile_action_mask(branches: ActionBranchesConfig) -> dict[str, tuple[int, ...]]:
    mask: dict[str, tuple[int, ...]] = {}
    for branch_name in ("gas", "air_brake", "boost", "lean"):
        branch = getattr(branches, branch_name)
        if branch is None:
            continue
        _validate_discrete_branch(branch_name, branch)
        mask_indices = _branch_mask_indices(branch_name, branch)
        if mask_indices is not None:
            mask[branch_name] = mask_indices
    return mask


def _validate_discrete_branch(branch_name: str, branch: ActionBranchConfig) -> None:
    if branch.type != "discrete":
        raise ValueError(f"action branch {branch_name!r} must be discrete")
    _validate_branch_specific_fields(branch_name, branch)


def _validate_branch_specific_fields(branch_name: str, branch: ActionBranchConfig) -> None:
    if branch.response_power is not None and branch_name != "steer":
        raise ValueError(f"action branch {branch_name!r} cannot define response_power")
    if branch.mode is not None and branch_name != "lean":
        raise ValueError(f"action branch {branch_name!r} cannot define mode")
    if branch.unmask_max_speed_kph is not None and branch_name != "boost":
        raise ValueError(f"action branch {branch_name!r} cannot define unmask_max_speed_kph")
    if branch.unmask_min_speed_kph is not None and branch_name != "lean":
        raise ValueError(f"action branch {branch_name!r} cannot define unmask_min_speed_kph")


def _branch_mask_indices(
    branch_name: str,
    branch: ActionBranchConfig,
) -> tuple[int, ...] | None:
    if branch.mask is None:
        return None
    expected_values = _DISCRETE_BRANCH_VALUES[branch_name]
    indices: list[int] = []
    for mask_value in branch.mask:
        try:
            indices.append(expected_values.index(mask_value))
        except ValueError as exc:
            raise ValueError(
                f"Unknown mask value {mask_value!r} for action branch {branch_name!r}"
            ) from exc
    return tuple(indices)
