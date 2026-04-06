# src/rl_fzerox/core/envs/defaults.py
from __future__ import annotations

import numpy as np
from gymnasium import spaces

from rl_fzerox.core.emulator.base import FrameStep


class NoOpActionSpec:
    def __init__(self) -> None:
        self._action_space = spaces.Discrete(1)

    @property
    def action_space(self) -> spaces.Discrete:
        return self._action_space

    def validate(self, action: int | np.integer) -> None:
        if int(action) != 0:
            raise ValueError(f"No-op action spec only accepts action 0, got {action!r}")


class RawFrameObservationSpec:
    def observation_space(self, frame_shape: tuple[int, int, int]) -> spaces.Box:
        return spaces.Box(low=0, high=255, shape=frame_shape, dtype=np.uint8)

    def transform(self, frame: np.ndarray) -> np.ndarray:
        return frame.copy()


class BackendReward:
    def reset(self) -> None:
        return None

    def reward(self, frame_step: FrameStep) -> float:
        return float(frame_step.reward)
