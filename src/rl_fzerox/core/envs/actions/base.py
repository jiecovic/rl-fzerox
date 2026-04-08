# src/rl_fzerox/core/envs/actions/base.py
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, TypeAlias

import numpy as np
from gymnasium import spaces

from rl_fzerox.core.emulator.control import ControllerState

ActionValue: TypeAlias = int | np.integer | Sequence[int] | np.ndarray


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
    def idle_action(self) -> np.ndarray:
        """Return the neutral action value for this adapter."""
        ...

    def decode(self, action: ActionValue) -> ControllerState:
        """Translate one policy action into a held controller state."""
        ...


def coerce_action_values(action: ActionValue) -> list[int]:
    """Normalize one policy action into a flat integer list."""

    if isinstance(action, np.ndarray):
        return action.astype(np.int64, copy=False).reshape(-1).tolist()
    if isinstance(action, np.integer):
        return [int(action)]
    if isinstance(action, Sequence) and not isinstance(action, str | bytes):
        return [int(value) for value in action]
    return [int(action)]


def steer_values(steer_buckets: int) -> np.ndarray:
    """Build evenly spaced steering levels spanning full left to full right."""

    return np.linspace(-1.0, 1.0, num=steer_buckets, dtype=np.float32)


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
            f"{action_label} actions must contain exactly {expected_size} values: "
            f"[{labels}]"
        )

    for value, dimension in zip(values, dimensions, strict=True):
        if not 0 <= value < dimension.size:
            raise ValueError(f"Invalid {dimension.label} index {value}")
    return tuple(values)
