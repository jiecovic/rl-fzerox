# tests/support/fakes.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from fzerox_emulator import (
    BackendStepResult,
    ControllerState,
    FrameStep,
    FZeroXTelemetry,
    ObservationSpec,
    ObservationStackMode,
    ResetState,
    StepStatus,
    StepSummary,
    display_size,
)
from fzerox_emulator.arrays import ObservationFrame, RgbFrame
from tests.support.native_objects import make_telemetry

_ObservationStackKey = tuple[str, int, ObservationStackMode, bool, object, object]


@dataclass
class SyntheticState:
    frame_index: int = 0
    progress: float = 0.0
    step_count: int = 0
    stalled_steps: int = 0
    reverse_timer: int = 0
    progress_frontier_stalled_frames: int = 0
    progress_frontier_distance: float = 0.0
    progress_frontier_initialized: bool = False


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
        self._observation_stacks: dict[_ObservationStackKey, tuple[list[RgbFrame], int | None]] = {}
        self.randomized_rng_seeds: list[int] = []
        self.loaded_baselines: list[Path] = []
        self.loaded_baseline_bytes: list[tuple[Path | None, int]] = []
        self._active_baseline_path: Path | None = None
        self._system_ram = bytearray(0x0030_0000)

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
                "baseline_kind": "custom" if self._active_baseline_path is not None else "startup",
                "baseline_state_path": (
                    None if self._active_baseline_path is None else str(self._active_baseline_path)
                ),
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

    def render(self) -> RgbFrame:
        return self._last_frame.copy()

    def observation_spec(self, preset: str) -> ObservationSpec:
        canonical_preset = _canonical_observation_preset(preset)
        if canonical_preset is None:
            raise ValueError(f"Unsupported synthetic observation preset {preset!r}")
        cropped = _crop_visible_game_area(self._last_frame)
        display_width, display_height = display_size(cropped.shape, self.display_aspect_ratio)
        if canonical_preset == "crop_84x116":
            width, height = (116, 84)
        elif canonical_preset == "crop_92x124":
            width, height = (124, 92)
        elif canonical_preset == "crop_116x164":
            width, height = (164, 116)
        elif canonical_preset == "crop_98x130":
            width, height = (130, 98)
        elif canonical_preset == "crop_66x82":
            width, height = (82, 66)
        elif canonical_preset == "crop_60x76":
            width, height = (76, 60)
        elif canonical_preset == "crop_68x68":
            width, height = (68, 68)
        elif canonical_preset == "crop_84x84":
            width, height = (84, 84)
        elif canonical_preset == "crop_76x100":
            width, height = (100, 76)
        else:
            width, height = (64, 64)
        return ObservationSpec(
            preset=canonical_preset,
            width=width,
            height=height,
            channels=3,
            display_width=display_width,
            display_height=display_height,
        )

    def render_display(self, *, preset: str) -> RgbFrame:
        spec = self.observation_spec(preset)
        cropped = _crop_visible_game_area(self._last_frame)
        aspect_corrected = _resize_frame(
            cropped,
            width=spec.display_width,
            height=spec.display_height,
        )
        return aspect_corrected

    def render_observation(
        self,
        *,
        preset: str,
        frame_stack: int,
        stack_mode: ObservationStackMode = "rgb",
        minimap_layer: bool = False,
        resize_filter: object = "nearest",
        minimap_resize_filter: object = "nearest",
    ) -> ObservationFrame:
        _ = (resize_filter, minimap_resize_filter)
        spec = self.observation_spec(preset)
        cropped = _crop_visible_game_area(self._last_frame)
        aspect_corrected = _resize_frame(
            cropped,
            width=spec.display_width,
            height=spec.display_height,
        )
        frame = _resize_frame(aspect_corrected, width=spec.width, height=spec.height)
        stack_key = (
            spec.preset,
            frame_stack,
            stack_mode,
            minimap_layer,
            resize_filter,
            minimap_resize_filter,
        )
        stacked_entry = self._observation_stacks.get(stack_key)
        if stacked_entry is None or stacked_entry[1] is None:
            frames = [np.array(frame, copy=True) for _ in range(frame_stack)]
            self._observation_stacks[stack_key] = (frames, self.frame_index)
            return _materialize_observation_stack(
                frames,
                stack_mode=stack_mode,
                minimap_layer=minimap_layer,
            )

        frames, last_frame_index = stacked_entry
        if last_frame_index != self.frame_index:
            frames = [*frames[1:], np.array(frame, copy=True)]
            self._observation_stacks[stack_key] = (frames, self.frame_index)
        return _materialize_observation_stack(
            self._observation_stacks[stack_key][0],
            stack_mode,
            minimap_layer=minimap_layer,
        )

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

    def read_system_ram(self, offset: int, length: int) -> bytes:
        return bytes(self._system_ram[offset : offset + length])

    def write_system_ram(self, offset: int, data: bytes) -> None:
        self._system_ram[offset : offset + len(data)] = data

    def vehicle_setup_info(self) -> dict[str, object]:
        return {}

    def step_repeat_raw(
        self,
        controller_state: ControllerState,
        *,
        action_repeat: int,
        preset: str,
        frame_stack: int,
        stack_mode: ObservationStackMode = "rgb",
        minimap_layer: bool = False,
        resize_filter: object = "nearest",
        minimap_resize_filter: object = "nearest",
        stuck_min_speed_kph: float,
        energy_loss_epsilon: float,
        max_episode_steps: int,
        progress_frontier_stall_limit_frames: int | None,
        progress_frontier_epsilon: float,
        terminate_on_energy_depleted: bool,
        lean_timer_assist: bool = False,
    ) -> BackendStepResult:
        _ = (
            stuck_min_speed_kph,
            energy_loss_epsilon,
            terminate_on_energy_depleted,
            lean_timer_assist,
        )
        self.set_controller_state(controller_state)
        if action_repeat <= 0:
            raise ValueError("action_repeat must be positive")

        self._capture_video_flags.extend([False] * max(action_repeat - 1, 0))
        self._capture_video_flags.append(True)
        for _ in range(action_repeat):
            self.step_frame()
        self._state.step_count += action_repeat
        if self._state.progress_frontier_initialized:
            frontier_reached = (
                self._state.progress
                >= self._state.progress_frontier_distance + progress_frontier_epsilon
            )
            if frontier_reached:
                self._state.progress_frontier_distance = self._state.progress
                self._state.progress_frontier_stalled_frames = 0
            else:
                self._state.progress_frontier_stalled_frames += action_repeat
        else:
            self._state.progress_frontier_distance = self._state.progress
            self._state.progress_frontier_initialized = True
            self._state.progress_frontier_stalled_frames = 0
        observation = self.render_observation(
            preset=preset,
            frame_stack=frame_stack,
            stack_mode=stack_mode,
            minimap_layer=minimap_layer,
            resize_filter=resize_filter,
            minimap_resize_filter=minimap_resize_filter,
        )
        truncation_reason = None
        if self._state.step_count >= max_episode_steps:
            truncation_reason = "timeout"
        elif (
            progress_frontier_stall_limit_frames is not None
            and self._state.progress_frontier_stalled_frames >= progress_frontier_stall_limit_frames
        ):
            truncation_reason = "progress_stalled"
        return BackendStepResult(
            observation=observation,
            summary=StepSummary(
                frames_run=action_repeat,
                max_race_distance=self._state.progress,
                reverse_active_frames=0,
                low_speed_frames=0,
                energy_loss_total=0.0,
                energy_gain_total=0.0,
                damage_taken_frames=0,
                consecutive_low_speed_frames=0,
                entered_state_flags=0,
                final_frame_index=self._state.frame_index,
            ),
            status=StepStatus(
                step_count=self._state.step_count,
                stalled_steps=self._state.stalled_steps,
                reverse_timer=self._state.reverse_timer,
                progress_frontier_stalled_frames=self._state.progress_frontier_stalled_frames,
                truncation_reason=truncation_reason,
            ),
            telemetry=None,
        )

    def step_repeat_watch_raw(
        self,
        controller_state: ControllerState,
        *,
        action_repeat: int,
        preset: str,
        frame_stack: int,
        stack_mode: ObservationStackMode = "rgb",
        minimap_layer: bool = False,
        resize_filter: object = "nearest",
        minimap_resize_filter: object = "nearest",
        stuck_min_speed_kph: float,
        energy_loss_epsilon: float,
        max_episode_steps: int,
        progress_frontier_stall_limit_frames: int | None,
        progress_frontier_epsilon: float,
        terminate_on_energy_depleted: bool,
        lean_timer_assist: bool = False,
    ) -> BackendStepResult:
        _ = (
            stuck_min_speed_kph,
            energy_loss_epsilon,
            terminate_on_energy_depleted,
            lean_timer_assist,
        )
        self.set_controller_state(controller_state)
        if action_repeat <= 0:
            raise ValueError("action_repeat must be positive")

        display_frames: list[RgbFrame] = []
        self._capture_video_flags.extend([True] * action_repeat)
        for _ in range(action_repeat):
            self.step_frame()
            display_frames.append(self.render_display(preset=preset))
        self._state.step_count += action_repeat
        if self._state.progress_frontier_initialized:
            frontier_reached = (
                self._state.progress
                >= self._state.progress_frontier_distance + progress_frontier_epsilon
            )
            if frontier_reached:
                self._state.progress_frontier_distance = self._state.progress
                self._state.progress_frontier_stalled_frames = 0
            else:
                self._state.progress_frontier_stalled_frames += action_repeat
        else:
            self._state.progress_frontier_distance = self._state.progress
            self._state.progress_frontier_initialized = True
            self._state.progress_frontier_stalled_frames = 0
        observation = self.render_observation(
            preset=preset,
            frame_stack=frame_stack,
            stack_mode=stack_mode,
            minimap_layer=minimap_layer,
            resize_filter=resize_filter,
            minimap_resize_filter=minimap_resize_filter,
        )
        truncation_reason = None
        if self._state.step_count >= max_episode_steps:
            truncation_reason = "timeout"
        elif (
            progress_frontier_stall_limit_frames is not None
            and self._state.progress_frontier_stalled_frames >= progress_frontier_stall_limit_frames
        ):
            truncation_reason = "progress_stalled"
        return BackendStepResult(
            observation=observation,
            summary=StepSummary(
                frames_run=action_repeat,
                max_race_distance=self._state.progress,
                reverse_active_frames=0,
                low_speed_frames=0,
                energy_loss_total=0.0,
                energy_gain_total=0.0,
                damage_taken_frames=0,
                consecutive_low_speed_frames=0,
                entered_state_flags=0,
                final_frame_index=self._state.frame_index,
            ),
            status=StepStatus(
                step_count=self._state.step_count,
                stalled_steps=self._state.stalled_steps,
                reverse_timer=self._state.reverse_timer,
                progress_frontier_stalled_frames=self._state.progress_frontier_stalled_frames,
                truncation_reason=truncation_reason,
            ),
            telemetry=None,
            display_frames=tuple(display_frames),
        )

    def set_controller_state(self, controller_state: ControllerState) -> None:
        self._last_controller_state = controller_state

    def load_baseline(self, path: Path) -> None:
        self._active_baseline_path = path
        self.loaded_baselines.append(path)

    def load_baseline_bytes(self, state: bytes, *, source_path: Path | None = None) -> None:
        self._active_baseline_path = source_path
        self.loaded_baseline_bytes.append((source_path, len(state)))

    def close(self) -> None:
        return None

    def _build_frame(self) -> RgbFrame:
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


def _canonical_observation_preset(preset: str) -> str | None:
    aliases = {
        "crop_84x116": "crop_84x116",
        "crop_92x124": "crop_92x124",
        "crop_116x164": "crop_116x164",
        "crop_98x130": "crop_98x130",
        "crop_66x82": "crop_66x82",
        "crop_60x76": "crop_60x76",
        "crop_68x68": "crop_68x68",
        "crop_84x84": "crop_84x84",
        "crop_76x100": "crop_76x100",
        "crop_64x64": "crop_64x64",
    }
    return aliases.get(preset)


def _crop_visible_game_area(frame: RgbFrame) -> RgbFrame:
    if frame.shape[0] <= 32 or frame.shape[1] <= 48:
        raise ValueError(f"Frame too small for visible game crop: {frame.shape!r}")
    return np.ascontiguousarray(frame[16:-16, 24:-24])


def _resize_frame(frame: RgbFrame, *, width: int, height: int) -> RgbFrame:
    input_height, input_width, _ = frame.shape
    if input_height == height and input_width == width:
        return np.array(frame, copy=True)

    y_index = np.rint(np.linspace(0, input_height - 1, num=height)).astype(np.intp)
    x_index = np.rint(np.linspace(0, input_width - 1, num=width)).astype(np.intp)
    return np.ascontiguousarray(frame[y_index][:, x_index])


def _materialize_observation_stack(
    frames: list[RgbFrame],
    stack_mode: ObservationStackMode,
    *,
    minimap_layer: bool = False,
) -> ObservationFrame:
    if not frames:
        raise ValueError("observation frame stack must not be empty")
    if stack_mode == "rgb":
        stacked = np.concatenate(frames, axis=2)
        return _append_fake_minimap_layer(stacked, frames[-1]) if minimap_layer else stacked
    if stack_mode == "gray":
        stacked = np.concatenate([_rgb_luma(frame) for frame in frames], axis=2)
        return _append_fake_minimap_layer(stacked, frames[-1]) if minimap_layer else stacked
    if stack_mode == "luma_chroma":
        stacked = np.concatenate([_rgb_luma_chroma(frame) for frame in frames], axis=2)
        return _append_fake_minimap_layer(stacked, frames[-1]) if minimap_layer else stacked
    raise ValueError(f"Unsupported observation stack mode: {stack_mode!r}")


def _append_fake_minimap_layer(
    observation: ObservationFrame,
    current_frame: RgbFrame,
) -> ObservationFrame:
    minimap = _rgb_luma(current_frame)
    return np.ascontiguousarray(np.concatenate([observation, minimap], axis=2))


def _rgb_luma(frame: RgbFrame) -> ObservationFrame:
    red = frame[:, :, 0].astype(np.uint16)
    green = frame[:, :, 1].astype(np.uint16)
    blue = frame[:, :, 2].astype(np.uint16)
    luma = ((77 * red) + (150 * green) + (29 * blue) + 128) >> 8
    return luma.astype(np.uint8)[:, :, None]


def _rgb_luma_chroma(frame: RgbFrame) -> ObservationFrame:
    red = frame[:, :, 0].astype(np.int16)
    green = frame[:, :, 1].astype(np.int16)
    blue = frame[:, :, 2].astype(np.int16)
    luma = _rgb_luma(frame)
    opponent = np.trunc(((2 * green) - red - blue) / 4.0).astype(np.int16)
    chroma = np.clip(128 + opponent, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(np.concatenate([luma, chroma[:, :, None]], axis=2))
