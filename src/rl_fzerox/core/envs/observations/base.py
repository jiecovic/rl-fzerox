# src/rl_fzerox/core/envs/observations/base.py
from __future__ import annotations

from typing import Protocol

import numpy as np
from gymnasium import spaces


class ObservationAdapter(Protocol):
    """Map raw emulator frames into policy observations."""

    @property
    def observation_space(self) -> spaces.Box:
        """Return the Gymnasium observation space for this adapter."""
        ...

    def transform(self, frame: np.ndarray, *, info: dict[str, object]) -> np.ndarray:
        """Convert one raw emulator frame into one observation."""
        ...
