# src/rl_fzerox/core/emulator/base.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from rl_fzerox.core.emulator.control import ControllerState


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
    """Emulator contract consumed by the F-Zero X env and watch tools."""

    @property
    def name(self) -> str: ...

    @property
    def native_fps(self) -> float: ...

    @property
    def frame_shape(self) -> tuple[int, int, int]: ...

    @property
    def frame_index(self) -> int: ...

    def reset(self) -> ResetState: ...

    def step_frame(self) -> FrameStep: ...

    def step_frames(self, count: int) -> None: ...

    def set_controller_state(self, controller_state: ControllerState) -> None: ...

    def render(self) -> np.ndarray: ...

    def close(self) -> None: ...
