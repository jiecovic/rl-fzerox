# src/rl_fzerox/core/envs/actions/base.py
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol, TypeAlias, runtime_checkable

import numpy as np
from gymnasium import spaces

from fzerox_emulator import ControllerState

ActionScalar: TypeAlias = int | float | np.integer | np.floating
ActionBranchValue: TypeAlias = ActionScalar | Sequence[ActionScalar] | np.ndarray
HybridActionValue: TypeAlias = Mapping[str, ActionBranchValue]
ActionValue: TypeAlias = ActionBranchValue | HybridActionValue


@dataclass(frozen=True, slots=True)
class DiscreteActionDimension:
    """One discrete action head with a human-readable validation label."""

    label: str
    size: int


class ActionAdapter(Protocol):
    """Map policy actions into held emulator controller state."""

    @property
    def action_space(self) -> spaces.Space:
        """Return the Gymnasium action space consumed by this adapter."""
        ...

    @property
    def idle_action(self) -> ActionValue:
        """Return the neutral action value for this adapter."""
        ...

    @property
    def action_dimensions(self) -> tuple[DiscreteActionDimension, ...]:
        """Return the ordered discrete action heads used by this adapter."""
        ...

    def decode(self, action: ActionValue) -> ControllerState:
        """Translate one policy action into a held controller state."""
        ...

    def action_mask(
        self,
        *,
        base_overrides: dict[str, tuple[int, ...]] | None = None,
        stage_overrides: dict[str, tuple[int, ...]] | None = None,
        dynamic_overrides: dict[str, tuple[int, ...]] | None = None,
    ) -> np.ndarray:
        """Return one flattened MultiDiscrete mask for this adapter."""
        ...


@runtime_checkable
class ResettableActionAdapter(Protocol):
    """Optional hook for action adapters with episode-local state."""

    def reset(self) -> None:
        """Clear per-episode action adapter state."""
        ...


def coerce_action_values(action: ActionValue) -> list[int]:
    """Normalize one policy action into a flat integer list."""

    if isinstance(action, Mapping):
        raise ValueError("Discrete actions must be a numeric scalar or sequence")
    if isinstance(action, np.ndarray):
        return action.astype(np.int64, copy=False).reshape(-1).tolist()
    if isinstance(action, np.integer):
        return [int(action)]
    if isinstance(action, Sequence) and not isinstance(action, str | bytes):
        return [int(value) for value in action]
    return [int(action)]


def shape_steer_value(steer: float, *, response_power: float) -> float:
    """Apply a sign-preserving response curve to normalized steering input."""

    clipped = float(np.clip(steer, -1.0, 1.0))
    if response_power == 1.0:
        return clipped
    return float(np.sign(clipped) * (abs(clipped) ** response_power))


def steer_values(steer_buckets: int, *, response_power: float = 1.0) -> np.ndarray:
    """Build steering levels spanning full left to full right."""

    values = np.linspace(-1.0, 1.0, num=steer_buckets, dtype=np.float32)
    if response_power == 1.0:
        return values
    shaped = np.sign(values) * (np.abs(values) ** float(response_power))
    return shaped.astype(np.float32, copy=False)


def multidiscrete_space(*sizes: int) -> spaces.MultiDiscrete:
    """Build one integer MultiDiscrete space with stable int64 storage."""

    return spaces.MultiDiscrete(np.array(sizes, dtype=np.int64))


def idle_action(*values: int) -> np.ndarray:
    """Build one neutral action value using the repo's int64 convention."""

    return np.array(values, dtype=np.int64)


def parse_discrete_action(
    action: ActionValue,
    *,
    action_label: str,
    dimensions: Sequence[DiscreteActionDimension],
) -> tuple[int, ...]:
    """Validate one discrete action vector against named bounded dimensions."""

    values = coerce_action_values(action)
    expected_size = len(dimensions)
    if len(values) != expected_size:
        labels = ", ".join(dimension.label for dimension in dimensions)
        raise ValueError(
            f"{action_label} actions must contain exactly {expected_size} values: [{labels}]"
        )

    for value, dimension in zip(values, dimensions, strict=True):
        if not 0 <= value < dimension.size:
            raise ValueError(f"Invalid {dimension.label} index {value}")
    return tuple(values)


def build_flat_action_mask(
    dimensions: tuple[DiscreteActionDimension, ...],
    *,
    base_overrides: dict[str, tuple[int, ...]] | None = None,
    stage_overrides: dict[str, tuple[int, ...]] | None = None,
    dynamic_overrides: dict[str, tuple[int, ...]] | None = None,
) -> np.ndarray:
    """Build one flattened boolean mask for a MultiDiscrete action space."""

    mask_values: list[bool] = []
    for dimension in dimensions:
        allowed_indices = _resolve_allowed_indices(
            dimension=dimension,
            base_overrides=base_overrides,
            stage_overrides=stage_overrides,
        )
        allowed_indices = _intersect_allowed_indices(
            dimension=dimension,
            allowed_indices=allowed_indices,
            dynamic_overrides=dynamic_overrides,
        )
        branch_mask = np.zeros(dimension.size, dtype=bool)
        if allowed_indices is None:
            branch_mask[:] = True
        else:
            for index in allowed_indices:
                if not 0 <= index < dimension.size:
                    raise ValueError(
                        f"Invalid masked {dimension.label} index {index}; "
                        f"valid range is [0, {dimension.size - 1}]"
                    )
                branch_mask[index] = True
            if not branch_mask.any():
                raise ValueError(f"Masked {dimension.label} branch must allow at least one action")
        mask_values.extend(branch_mask.tolist())
    return np.asarray(mask_values, dtype=bool)


def _resolve_allowed_indices(
    *,
    dimension: DiscreteActionDimension,
    base_overrides: dict[str, tuple[int, ...]] | None,
    stage_overrides: dict[str, tuple[int, ...]] | None,
) -> tuple[int, ...] | None:
    if stage_overrides is not None and dimension.label in stage_overrides:
        return stage_overrides[dimension.label]
    if base_overrides is not None and dimension.label in base_overrides:
        return base_overrides[dimension.label]
    return None


def _intersect_allowed_indices(
    *,
    dimension: DiscreteActionDimension,
    allowed_indices: tuple[int, ...] | None,
    dynamic_overrides: dict[str, tuple[int, ...]] | None,
) -> tuple[int, ...] | None:
    if dynamic_overrides is None or dimension.label not in dynamic_overrides:
        return allowed_indices
    dynamic_indices = dynamic_overrides[dimension.label]
    if allowed_indices is None:
        return dynamic_indices
    allowed_set = set(allowed_indices)
    return tuple(index for index in dynamic_indices if index in allowed_set)
