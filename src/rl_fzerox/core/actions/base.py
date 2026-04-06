# src/rl_fzerox/core/actions/base.py
from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, TypeAlias

import numpy as np
from gymnasium import spaces

from rl_fzerox.core.emulator.control import ControllerState

ActionValue: TypeAlias = int | np.integer | Sequence[int] | np.ndarray


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
