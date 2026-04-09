# tests/support/fakes.py
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fzerox_emulator import (
    BackendStepResult,
    ControllerState,
    FrameStep,
    FZeroXTelemetry,
    ObservationSpec,
    ResetState,
    StepStatus,
    StepSummary,
    display_size,
)
from tests.support.native_objects import make_telemetry


@dataclass
class SyntheticState:
    frame_index: int = 0
    progress: float = 0.0
    step_count: int = 0
    stalled_steps: int = 0
    reverse_timer: int = 0


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
        self.randomized_rng_seeds: list[int] = []

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
        if preset not in {"native_crop_v1", "native_crop_v2", "native_crop_v3"}:
            raise ValueError(f"Unsupported synthetic observation preset {preset!r}")
        cropped = _crop_native_crop_v1(self._last_frame)
        display_width, display_height = display_size(cropped.shape, self.display_aspect_ratio)
        if preset == "native_crop_v1":
            width, height = (116, 84)
        elif preset == "native_crop_v2":
            width, height = (124, 92)
        else:
            width, height = (164, 116)
        return ObservationSpec(
            preset=preset,
            width=width,
            height=height,
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
        aspect_corrected = _resize_frame(
            cropped,
            width=spec.display_width,
            height=spec.display_height,
        )
        frame = _resize_frame(aspect_corrected, width=spec.width, height=spec.height)
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
        if self.frame_index < 240:
            return None
        if self.frame_index < 1_412:
            return make_telemetry(
                game_mode_raw=0,
                game_mode_name="title",
                in_race_mode=False,
                race_distance=self._state.progress,
                state_labels=("active",),
            )
        return make_telemetry(
            game_mode_raw=1,
            game_mode_name="gp_race",
            in_race_mode=True,
            race_distance=self._state.progress,
            race_time_ms=0,
            state_labels=("active",),
        )

    def step_frames(self, count: int, *, capture_video: bool = True) -> None:
        self._capture_video_flags.extend([capture_video] * count)
        for _ in range(count):
            self.step_frame()

    def randomize_game_rng(self, seed: int) -> tuple[int, int, int, int]:
        self.randomized_rng_seeds.append(seed)
        return (
            seed & 0xFFFF_FFFF,
            (seed >> 32) & 0xFFFF_FFFF,
            (seed ^ 0xA5A5_A5A5) & 0xFFFF_FFFF,
            ((seed >> 32) ^ 0x5A5A_5A5A) & 0xFFFF_FFFF,
        )

    def step_repeat_raw(
        self,
        controller_state: ControllerState,
        *,
        action_repeat: int,
        preset: str,
        frame_stack: int,
        stuck_min_speed_kph: float,
        energy_loss_epsilon: float,
        max_episode_steps: int,
        stuck_step_limit: int,
        wrong_way_timer_limit: int,
    ) -> BackendStepResult:
        _ = (
            stuck_min_speed_kph,
            energy_loss_epsilon,
            wrong_way_timer_limit,
        )
        self.set_controller_state(controller_state)
        if action_repeat <= 0:
            raise ValueError("action_repeat must be positive")

        self._capture_video_flags.extend([False] * max(action_repeat - 1, 0))
        self._capture_video_flags.append(True)
        for _ in range(action_repeat):
            self.step_frame()
        self._state.step_count += action_repeat
        observation = self.render_observation(preset=preset, frame_stack=frame_stack)
        truncation_reason = None
        if self._state.step_count >= max_episode_steps:
            truncation_reason = "timeout"
        return BackendStepResult(
            observation=observation,
            summary=StepSummary(
                frames_run=action_repeat,
                max_race_distance=self._state.progress,
                reverse_active_frames=0,
                low_speed_frames=0,
                energy_loss_total=0.0,
                energy_gain_total=0.0,
                consecutive_low_speed_frames=0,
                entered_state_flags=0,
                final_frame_index=self._state.frame_index,
            ),
            status=StepStatus(
                step_count=self._state.step_count,
                stalled_steps=self._state.stalled_steps,
                reverse_timer=self._state.reverse_timer,
                truncation_reason=truncation_reason,
            ),
            telemetry=None,
        )

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
