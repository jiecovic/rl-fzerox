# src/fzerox_emulator/base.py
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Protocol, TypeAlias

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
    """Resolved native observation geometry for one preset."""

    preset: str
    width: int
    height: int
    channels: int
    display_width: int
    display_height: int


ObservationStackMode: TypeAlias = Literal["rgb", "rgb_gray"]


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
    if stack_mode == "rgb_gray":
        if frame_stack == 1:
            return single_frame_channels + extra_channels
        return (frame_stack - 1) + single_frame_channels + extra_channels
    raise ValueError(f"Unsupported observation stack mode: {stack_mode!r}")


@dataclass(frozen=True)
class BackendStepResult:
    """Native repeated-step payload consumed by the env engine."""

    observation: ObservationFrame
    summary: StepSummary
    status: StepStatus
    telemetry: FZeroXTelemetry | None
    display_frames: tuple[RgbFrame, ...] = ()


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

    def step_repeat_raw(
        self,
        controller_state: ControllerState,
        *,
        action_repeat: int,
        preset: str,
        frame_stack: int,
        stack_mode: ObservationStackMode = "rgb",
        minimap_layer: bool = False,
        stuck_min_speed_kph: float,
        energy_loss_epsilon: float,
        max_episode_steps: int,
        stuck_step_limit: int,
        wrong_way_timer_limit: int | None,
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
        preset: str,
        frame_stack: int,
        stack_mode: ObservationStackMode = "rgb",
        minimap_layer: bool = False,
        stuck_min_speed_kph: float,
        energy_loss_epsilon: float,
        max_episode_steps: int,
        stuck_step_limit: int,
        wrong_way_timer_limit: int | None,
        progress_frontier_stall_limit_frames: int | None,
        progress_frontier_epsilon: float,
        terminate_on_energy_depleted: bool,
        lean_timer_assist: bool = False,
    ) -> BackendStepResult: ...

    def set_controller_state(self, controller_state: ControllerState) -> None: ...

    def load_baseline(self, path: Path) -> None: ...

    def load_baseline_bytes(self, state: bytes, *, source_path: Path | None = None) -> None: ...

    def render(self) -> RgbFrame: ...

    def observation_spec(self, preset: str) -> ObservationSpec: ...

    def render_display(self, *, preset: str) -> RgbFrame: ...

    def render_observation(
        self,
        *,
        preset: str,
        frame_stack: int,
        stack_mode: ObservationStackMode = "rgb",
        minimap_layer: bool = False,
    ) -> ObservationFrame: ...

    def try_read_telemetry(self) -> FZeroXTelemetry | None: ...

    def close(self) -> None: ...
