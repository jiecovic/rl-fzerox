# tests/fakes.py
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rl_fzerox.core.emulator.base import FrameStep, ResetState
from rl_fzerox.core.emulator.control import ControllerState


@dataclass
class SyntheticState:
    frame_index: int = 0
    progress: float = 0.0


class SyntheticBackend:
    def __init__(
        self,
        width: int = 160,
        height: int = 120,
        max_frames: int = 1_200,
    ):
        self._width = width
        self._height = height
        self._max_frames = max_frames
        self._state = SyntheticState()
        self._last_frame = self._build_frame()
        self._last_controller_state = ControllerState()

    @property
    def name(self) -> str:
        return "synthetic"

    @property
    def native_fps(self) -> float:
        return 60.0

    @property
    def frame_shape(self) -> tuple[int, int, int]:
        return (self._height, self._width, 3)

    @property
    def frame_index(self) -> int:
        return self._state.frame_index

    @property
    def last_controller_state(self) -> ControllerState:
        return self._last_controller_state

    def reset(self) -> ResetState:
        self._state = SyntheticState()
        self._last_frame = self._build_frame()
        self._last_controller_state = ControllerState()
        return ResetState(
            frame=self._last_frame,
            info={
                "backend": self.name,
                "native_fps": self.native_fps,
                "frame_index": self._state.frame_index,
                "baseline_kind": "startup",
                "progress": self._state.progress,
            },
        )

    def step_frame(self) -> FrameStep:
        self._state.frame_index += 1
        self._state.progress += 6.0
        self._last_frame = self._build_frame()
        return FrameStep(
            frame=self._last_frame,
            reward=0.0,
            terminated=False,
            truncated=self._state.frame_index >= self._max_frames,
            info={
                "backend": self.name,
                "frame_index": self._state.frame_index,
                "progress": self._state.progress,
            },
        )

    def render(self) -> np.ndarray:
        return self._last_frame.copy()

    def step_frames(self, count: int) -> None:
        for _ in range(count):
            self.step_frame()

    def set_controller_state(self, controller_state: ControllerState) -> None:
        self._last_controller_state = controller_state

    def close(self) -> None:
        return None

    def _build_frame(self) -> np.ndarray:
        frame = np.zeros((self._height, self._width, 3), dtype=np.uint8)
        frame[:, :, 0] = 32
        frame[:, :, 1] = 40
        frame[:, :, 2] = 54

        stripe_x = int((self._state.frame_index * 3) % self._width)
        frame[:, max(0, stripe_x - 2) : min(self._width, stripe_x + 2), :] = np.array(
            [240, 120, 72],
            dtype=np.uint8,
        )

        progress_width = int(self._state.progress % self._width)
        frame[8:14, :progress_width, :] = np.array([96, 220, 124], dtype=np.uint8)
        return frame
