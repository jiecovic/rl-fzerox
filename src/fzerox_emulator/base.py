# src/fzerox_emulator/base.py
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Protocol, TypeAlias, TypedDict

from fzerox_emulator._native import FZeroXTelemetry, StepStatus, StepSummary
from fzerox_emulator.arrays import ObservationFrame, RgbFrame
from fzerox_emulator.control import ControllerState


@dataclass(frozen=True)
class ResetState:
    """State returned immediately after an emulator reset."""

    frame: RgbFrame
    info: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class FrameStep:
    """One emulator frame worth of output and metadata."""

    frame: RgbFrame
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class ObservationSpec:
    """Resolved native observation geometry for one image layout."""

    preset: str
    width: int
    height: int
    channels: int
    display_width: int
    display_height: int


@dataclass(frozen=True, slots=True)
class ObservationImageRecipe:
    """One native image-render recipe for an observation view."""

    preset: str | None = None
    height: int | None = None
    width: int | None = None
    frame_stack: int = 1
    stack_mode: ObservationStackMode = "rgb"
    minimap_layer: bool = False
    resize_filter: ObservationResizeFilter = "nearest"
    minimap_resize_filter: ObservationResizeFilter = "nearest"

    def normalized_resolution(self) -> tuple[str | None, int | None, int | None]:
        return normalize_observation_resolution(
            preset=self.preset,
            height=self.height,
            width=self.width,
        )


ObservationStackMode: TypeAlias = Literal["rgb", "gray", "luma_chroma"]
ObservationResizeFilter: TypeAlias = Literal["nearest", "bilinear"]
RaceStartMode: TypeAlias = Literal["time_attack", "gp_race"]


class FrameObservationOptions(TypedDict, total=False):
    """Typed Python-side options passed into the native observation renderer."""

    stack_mode: ObservationStackMode
    minimap_layer: bool
    resize_filter: ObservationResizeFilter
    minimap_resize_filter: ObservationResizeFilter
    height: int
    width: int


def normalize_observation_resolution(
    *,
    preset: str | None = None,
    height: int | None = None,
    width: int | None = None,
) -> tuple[str | None, int | None, int | None]:
    """Return one validated preset-or-custom resolution triple."""

    if preset is not None:
        if height is not None or width is not None:
            raise ValueError("preset cannot be combined with custom observation height/width")
        return preset, None, None
    if height is None or width is None:
        raise ValueError("custom observation height and width must both be set")
    return None, int(height), int(width)


def stacked_observation_channels(
    single_frame_channels: int,
    *,
    frame_stack: int,
    stack_mode: ObservationStackMode,
    minimap_layer: bool = False,
) -> int:
    """Return channel count after temporal frame-stack encoding."""

    if frame_stack <= 0:
        raise ValueError("frame_stack must be positive")
    extra_channels = 1 if minimap_layer else 0
    if stack_mode == "rgb":
        return (single_frame_channels * frame_stack) + extra_channels
    if stack_mode == "gray":
        return frame_stack + extra_channels
    if stack_mode == "luma_chroma":
        return (frame_stack * 2) + extra_channels
    raise ValueError(f"Unsupported observation stack mode: {stack_mode!r}")


@dataclass(frozen=True)
class BackendStepResult:
    """Native repeated-step payload consumed by the env engine."""

    observation: ObservationFrame
    summary: StepSummary
    status: StepStatus
    telemetry: FZeroXTelemetry | None
    display_frames: tuple[RgbFrame, ...] = ()


@dataclass(frozen=True)
class BackendMultiObservationStepResult:
    """Native repeated-step payload with multiple rendered observation views."""

    observations: tuple[ObservationFrame, ...]
    summary: StepSummary
    status: StepStatus
    telemetry: FZeroXTelemetry | None


class EmulatorBackend(Protocol):
    """Emulator contract consumed by the F-Zero X env and watch tools."""

    @property
    def name(self) -> str: ...

    @property
    def native_fps(self) -> float: ...

    @property
    def display_aspect_ratio(self) -> float: ...

    @property
    def frame_shape(self) -> tuple[int, int, int]: ...

    @property
    def frame_index(self) -> int: ...

    def reset(self) -> ResetState: ...

    def step_frame(self) -> FrameStep: ...

    def step_frames(self, count: int, *, capture_video: bool = True) -> None: ...

    def randomize_game_rng(self, seed: int) -> tuple[int, int, int, int]: ...

    def read_system_ram(self, offset: int, length: int) -> bytes: ...

    def write_system_ram(self, offset: int, data: bytes) -> None: ...

    def vehicle_setup_info(self) -> dict[str, object]: ...

    def step_repeat_raw(
        self,
        controller_state: ControllerState,
        *,
        action_repeat: int,
        preset: str | None = None,
        height: int | None = None,
        width: int | None = None,
        frame_stack: int,
        stack_mode: ObservationStackMode = "rgb",
        minimap_layer: bool = False,
        resize_filter: ObservationResizeFilter = "nearest",
        minimap_resize_filter: ObservationResizeFilter = "nearest",
        stuck_min_speed_kph: float,
        energy_loss_epsilon: float,
        max_episode_steps: int,
        progress_frontier_stall_limit_frames: int | None,
        progress_frontier_epsilon: float,
        terminate_on_energy_depleted: bool,
        lean_timer_assist: bool = False,
    ) -> BackendStepResult: ...

    def step_repeat_watch_raw(
        self,
        controller_state: ControllerState,
        *,
        action_repeat: int,
        preset: str | None = None,
        height: int | None = None,
        width: int | None = None,
        frame_stack: int,
        stack_mode: ObservationStackMode = "rgb",
        minimap_layer: bool = False,
        resize_filter: ObservationResizeFilter = "nearest",
        minimap_resize_filter: ObservationResizeFilter = "nearest",
        stuck_min_speed_kph: float,
        energy_loss_epsilon: float,
        max_episode_steps: int,
        progress_frontier_stall_limit_frames: int | None,
        progress_frontier_epsilon: float,
        terminate_on_energy_depleted: bool,
        lean_timer_assist: bool = False,
    ) -> BackendStepResult: ...

    def step_repeat_multi_observation_raw(
        self,
        controller_state: ControllerState,
        *,
        action_repeat: int,
        observation_recipes: Sequence[ObservationImageRecipe],
        stuck_min_speed_kph: float,
        energy_loss_epsilon: float,
        max_episode_steps: int,
        progress_frontier_stall_limit_frames: int | None,
        progress_frontier_epsilon: float,
        terminate_on_energy_depleted: bool,
        lean_timer_assist: bool = False,
    ) -> BackendMultiObservationStepResult: ...

    def set_controller_state(self, controller_state: ControllerState) -> None: ...

    def load_baseline(self, path: Path) -> None: ...

    def load_baseline_bytes(self, state: bytes, *, source_path: Path | None = None) -> None: ...

    def patch_race_start_setup(
        self,
        *,
        mode: RaceStartMode,
        course_index: int,
        character_index: int,
        engine_setting_raw_value: int,
        total_lap_count: int,
    ) -> None: ...

    def patch_machine_settings(
        self,
        *,
        mode: RaceStartMode,
        course_index: int,
        character_index: int,
        engine_setting_raw_value: int,
        total_lap_count: int,
    ) -> None: ...

    def patch_time_attack_menu_mode(self) -> None: ...

    def patch_engine_settings(
        self,
        *,
        mode: RaceStartMode,
        engine_setting_raw_value: int,
    ) -> None: ...

    def force_race_reinit(self, *, mode: RaceStartMode) -> None: ...

    def validate_race_start_setup(
        self,
        *,
        mode: RaceStartMode,
        course_index: int,
        character_index: int,
        engine_setting_raw_value: int,
        total_lap_count: int,
    ) -> None: ...

    def patch_time_attack_race_start_setup(
        self,
        *,
        course_index: int,
        character_index: int,
        engine_setting_raw_value: int,
        total_lap_count: int,
    ) -> None: ...

    def patch_time_attack_machine_settings(
        self,
        *,
        course_index: int,
        character_index: int,
        engine_setting_raw_value: int,
        total_lap_count: int,
    ) -> None: ...

    def force_time_attack_reinit(self) -> None: ...

    def render(self) -> RgbFrame: ...

    def observation_spec(
        self,
        preset: str | None = None,
        *,
        height: int | None = None,
        width: int | None = None,
    ) -> ObservationSpec: ...

    def render_display(
        self,
        *,
        preset: str | None = None,
        height: int | None = None,
        width: int | None = None,
    ) -> RgbFrame: ...

    def render_observation(
        self,
        *,
        preset: str | None = None,
        height: int | None = None,
        width: int | None = None,
        frame_stack: int,
        stack_mode: ObservationStackMode = "rgb",
        minimap_layer: bool = False,
        resize_filter: ObservationResizeFilter = "nearest",
        minimap_resize_filter: ObservationResizeFilter = "nearest",
    ) -> ObservationFrame: ...

    def try_read_telemetry(self) -> FZeroXTelemetry | None: ...

    def close(self) -> None: ...
