# src/rl_fzerox/core/envs/contracts.py
from __future__ import annotations

from typing import Protocol

import numpy as np
from gymnasium.spaces import Space

from rl_fzerox.core.emulator.base import FrameStep


class ActionSpec(Protocol):
    @property
    def action_space(self) -> Space[np.int64]: ...

    def validate(self, action: int | np.integer) -> None: ...


class ObservationSpec(Protocol):
    def observation_space(self, frame_shape: tuple[int, int, int]) -> Space[np.ndarray]: ...

    def transform(self, frame: np.ndarray) -> np.ndarray: ...


class RewardFunction(Protocol):
    def reset(self) -> None: ...

    def reward(self, frame_step: FrameStep) -> float: ...
