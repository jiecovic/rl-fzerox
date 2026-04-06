# src/rl_fzerox/core/emulator/emulator.py
from __future__ import annotations

from pathlib import Path

import numpy as np

from rl_fzerox._native import Emulator as NativeEmulator
from rl_fzerox.core.emulator.base import FrameStep, ResetState
from rl_fzerox.core.emulator.control import ControllerState
from rl_fzerox.core.emulator.video import display_size


class Emulator:
    """Python wrapper over the native Rust libretro host."""

    def __init__(
        self,
        *,
        core_path: Path,
        rom_path: Path,
        runtime_dir: Path | None = None,
        baseline_state_path: Path | None = None,
    ) -> None:
        self._core_path = core_path.resolve()
        self._rom_path = rom_path.resolve()
        self._runtime_dir = runtime_dir.resolve() if runtime_dir is not None else None
        self._baseline_state_path = (
            baseline_state_path.resolve()
            if baseline_state_path is not None
            else None
        )
        self._native = NativeEmulator(
            str(self._core_path),
            str(self._rom_path),
            None if self._runtime_dir is None else str(self._runtime_dir),
            (
                None
                if self._baseline_state_path is None
                else str(self._baseline_state_path)
            ),
        )

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
        return display_size(self.frame_shape, self.display_aspect_ratio)

    @property
    def frame_index(self) -> int:
        return int(self._native.frame_index)

    @property
    def system_ram_size(self) -> int:
        return int(self._native.system_ram_size)

    @property
    def baseline_kind(self) -> str:
        return str(self._native.baseline_kind)

    def reset(self) -> ResetState:
        """Restore the deterministic episode baseline and return the first frame."""

        self._native.reset()
        return ResetState(
            frame=self.render(),
            info=self._frame_info(),
        )

    def step_frame(self) -> FrameStep:
        """Advance exactly one emulator frame."""

        self.step_frames(1)
        return FrameStep(
            frame=self.render(),
            reward=0.0,
            terminated=False,
            truncated=False,
            info=self._frame_info(),
        )

    def step_frames(self, count: int) -> None:
        """Advance the emulator by a fixed number of frames."""

        self._native.step_frames(count)

    def set_controller_state(self, controller_state: ControllerState) -> None:
        """Set the held controller state used for subsequent frame stepping."""

        state = controller_state.clamped()
        self._native.set_controller_state(
            joypad_mask=state.joypad_mask,
            left_stick_x=state.left_stick_x,
            left_stick_y=state.left_stick_y,
            right_stick_x=state.right_stick_x,
            right_stick_y=state.right_stick_y,
        )

    def save_state(self, path: Path) -> None:
        """Serialize the current emulator state to a savestate file."""

        self._native.save_state(str(path.resolve()))

    def read_system_ram(self, offset: int, length: int) -> bytes:
        """Read a raw slice from the libretro system RAM buffer."""

        return bytes(self._native.read_system_ram(offset, length))

    def capture_current_as_baseline(self, path: Path | None = None) -> None:
        """Promote the current state to the active reset baseline."""

        resolved_path = None if path is None else str(path.resolve())
        self._native.capture_current_as_baseline(resolved_path)

    def render(self) -> np.ndarray:
        """Return the latest raw RGB frame as a NumPy array."""

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
        """Release the native emulator host."""

        self._native.close()

    def _frame_info(self) -> dict[str, object]:
        return {
            "backend": self.name,
            "frame_index": self.frame_index,
            "core_path": str(self._core_path),
            "rom_path": str(self._rom_path),
            "runtime_dir": None if self._runtime_dir is None else str(self._runtime_dir),
            "baseline_state_path": (
                None
                if self._baseline_state_path is None
                else str(self._baseline_state_path)
            ),
            "baseline_kind": self.baseline_kind,
            "display_aspect_ratio": self.display_aspect_ratio,
            "native_fps": self.native_fps,
        }
