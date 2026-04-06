# src/rl_fzerox/core/emulator/base.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class ResetState:
    frame: np.ndarray
    info: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class FrameStep:
    frame: np.ndarray
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, object] = field(default_factory=dict)


class EmulatorBackend(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def native_fps(self) -> float: ...

    @property
    def frame_shape(self) -> tuple[int, int, int]: ...

    def reset(self, seed: int | None = None) -> ResetState: ...

    def step_frame(self) -> FrameStep: ...

    def render(self) -> np.ndarray: ...

    def close(self) -> None: ...
