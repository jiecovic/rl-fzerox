# src/rl_fzerox/core/emulator/base.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class ResetState:
    """State returned immediately after an emulator reset."""

    frame: np.ndarray
    info: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class FrameStep:
    """One emulator frame worth of output and metadata."""

    frame: np.ndarray
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, object] = field(default_factory=dict)


class EmulatorBackend(Protocol):
    """Minimal emulator contract consumed by the env and viewer."""

    @property
    def name(self) -> str: ...

    @property
    def native_fps(self) -> float: ...

    @property
    def frame_shape(self) -> tuple[int, int, int]: ...

    def reset(self) -> ResetState: ...

    def step_frame(self) -> FrameStep: ...

    def render(self) -> np.ndarray: ...

    def close(self) -> None: ...
