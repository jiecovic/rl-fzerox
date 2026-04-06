# src/rl_fzerox/core/emulator/emulator.py
from __future__ import annotations

from pathlib import Path

import numpy as np

from rl_fzerox._native import Emulator as NativeEmulator
from rl_fzerox.core.emulator.base import FrameStep, ResetState


class Emulator:
    def __init__(self, *, core_path: Path, rom_path: Path) -> None:
        self._core_path = core_path.resolve()
        self._rom_path = rom_path.resolve()
        self._native = NativeEmulator(str(self._core_path), str(self._rom_path))

    @property
    def name(self) -> str:
        return self._native.name

    @property
    def native_fps(self) -> float:
        return float(self._native.native_fps)

    @property
    def display_aspect_ratio(self) -> float:
        return float(self._native.display_aspect_ratio)

    @property
    def frame_shape(self) -> tuple[int, int, int]:
        height, width, channels = self._native.frame_shape
        return int(height), int(width), int(channels)

    @property
    def display_size(self) -> tuple[int, int]:
        frame_height, frame_width, _ = self.frame_shape
        if self.display_aspect_ratio <= 0.0:
            return frame_width, frame_height

        display_height = max(1, round(frame_width / self.display_aspect_ratio))
        return frame_width, int(display_height)

    @property
    def frame_index(self) -> int:
        return int(self._native.frame_index)

    def reset(self, seed: int | None = None) -> ResetState:
        _ = seed
        self._native.reset()
        return ResetState(
            frame=self.render(),
            info=self._frame_info(),
        )

    def step_frame(self) -> FrameStep:
        self.step_frames(1)
        return FrameStep(
            frame=self.render(),
            reward=0.0,
            terminated=False,
            truncated=False,
            info=self._frame_info(),
        )

    def step_frames(self, count: int) -> None:
        self._native.step_frames(count)

    def render(self) -> np.ndarray:
        frame_bytes = self._native.frame_rgb()
        frame_height, frame_width, channels = self.frame_shape
        frame = np.frombuffer(frame_bytes, dtype=np.uint8)
        expected_size = frame_height * frame_width * channels
        if frame.size != expected_size:
            raise RuntimeError(
                "Unexpected frame size from native emulator: "
                f"expected {expected_size} bytes, got {frame.size}"
            )
        return frame.reshape((frame_height, frame_width, channels))

    def close(self) -> None:
        self._native.close()

    def _frame_info(self) -> dict[str, object]:
        return {
            "backend": self.name,
            "runtime": "libretro",
            "frame_index": self.frame_index,
            "core_path": str(self._core_path),
            "rom_path": str(self._rom_path),
            "display_aspect_ratio": self.display_aspect_ratio,
            "native_fps": self.native_fps,
        }
