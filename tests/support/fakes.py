# tests/support/fakes.py
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rl_fzerox.core.emulator.base import FrameStep, ObservationSpec, ResetState
from rl_fzerox.core.emulator.control import ControllerState
from rl_fzerox.core.emulator.video import display_size
from rl_fzerox.core.game.telemetry import FZeroXTelemetry


@dataclass
class SyntheticState:
    frame_index: int = 0
    progress: float = 0.0


class SyntheticBackend:
    def __init__(
        self,
        width: int = 640,
        height: int = 240,
        max_frames: int = 1_200,
    ):
        self._width = width
        self._height = height
        self._max_frames = max_frames
        self._state = SyntheticState()
        self._last_frame = self._build_frame()
        self._last_controller_state = ControllerState()
        self._capture_video_flags: list[bool] = []
        self._observation_stacks: dict[tuple[str, int], tuple[np.ndarray, int | None]] = {}

    @property
    def name(self) -> str:
        return "synthetic"

    @property
    def native_fps(self) -> float:
        return 60.0

    @property
    def display_aspect_ratio(self) -> float:
        return 4.0 / 3.0

    @property
    def frame_shape(self) -> tuple[int, int, int]:
        return (self._height, self._width, 3)

    @property
    def frame_index(self) -> int:
        return self._state.frame_index

    @property
    def last_controller_state(self) -> ControllerState:
        return self._last_controller_state

    @property
    def capture_video_flags(self) -> list[bool]:
        return list(self._capture_video_flags)

    def reset(self) -> ResetState:
        self._state = SyntheticState()
        self._last_frame = self._build_frame()
        self._last_controller_state = ControllerState()
        self._capture_video_flags.clear()
        self._observation_stacks.clear()
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

    def observation_spec(self, preset: str) -> ObservationSpec:
        if preset != "native_crop_v1":
            raise ValueError(f"Unsupported synthetic observation preset {preset!r}")
        cropped = _crop_native_crop_v1(self._last_frame)
        display_width, display_height = display_size(cropped.shape, self.display_aspect_ratio)
        return ObservationSpec(
            preset=preset,
            width=222,
            height=78,
            channels=3,
            display_width=display_width,
            display_height=display_height,
        )

    def render_display(self, *, preset: str) -> np.ndarray:
        spec = self.observation_spec(preset)
        cropped = _crop_native_crop_v1(self._last_frame)
        aspect_corrected = _resize_frame(
            cropped,
            width=spec.display_width,
            height=spec.display_height,
        )
        return aspect_corrected

    def render_observation(self, *, preset: str, frame_stack: int) -> np.ndarray:
        spec = self.observation_spec(preset)
        cropped = _crop_native_crop_v1(self._last_frame)
        frame = _resize_frame(cropped, width=spec.width, height=spec.height)
        stack_key = (preset, frame_stack)
        stacked_entry = self._observation_stacks.get(stack_key)
        if stacked_entry is None or stacked_entry[1] is None:
            stacked = np.concatenate([frame] * frame_stack, axis=2)
            self._observation_stacks[stack_key] = (stacked, self.frame_index)
            return np.array(stacked, copy=True)

        stacked, last_frame_index = stacked_entry
        if last_frame_index != self.frame_index:
            channels = spec.channels
            stacked[:, :, :-channels] = stacked[:, :, channels:]
            stacked[:, :, -channels:] = frame
            self._observation_stacks[stack_key] = (stacked, self.frame_index)
        return np.array(self._observation_stacks[stack_key][0], copy=True)

    def try_read_telemetry(self) -> FZeroXTelemetry | None:
        return None

    def step_frames(self, count: int, *, capture_video: bool = True) -> None:
        self._capture_video_flags.extend([capture_video] * count)
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


def _crop_native_crop_v1(frame: np.ndarray) -> np.ndarray:
    if frame.shape[0] <= 32 or frame.shape[1] <= 48:
        raise ValueError(f"Frame too small for native_crop_v1: {frame.shape!r}")
    return np.ascontiguousarray(frame[16:-16, 24:-24])


def _resize_frame(frame: np.ndarray, *, width: int, height: int) -> np.ndarray:
    input_height, input_width, _ = frame.shape
    if input_height == height and input_width == width:
        return np.array(frame, copy=True)

    y_index = np.rint(np.linspace(0, input_height - 1, num=height)).astype(np.intp)
    x_index = np.rint(np.linspace(0, input_width - 1, num=width)).astype(np.intp)
    return np.ascontiguousarray(frame[y_index][:, x_index])
