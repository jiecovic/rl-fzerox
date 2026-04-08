# src/rl_fzerox/core/emulator/base.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

import numpy as np

from rl_fzerox.core.emulator.control import ControllerState

if TYPE_CHECKING:
    from rl_fzerox.core.game.telemetry import FZeroXTelemetry


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


@dataclass(frozen=True)
class ObservationSpec:
    """Resolved native observation geometry for one preset."""

    preset: str
    width: int
    height: int
    channels: int
    display_width: int
    display_height: int


@dataclass(frozen=True, slots=True)
class StepSummary:
    """Step-level aggregates returned after one repeated env step."""

    frames_run: int
    max_race_distance: float
    reverse_progress_total: float
    consecutive_reverse_frames: int
    energy_loss_total: float
    consecutive_low_speed_frames: int
    entered_state_flags: int
    final_frame_index: int


@dataclass(frozen=True)
class BackendStepResult:
    """Native repeated-step payload consumed by the env engine."""

    observation: np.ndarray
    summary: StepSummary
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

    def step_repeat_raw(
        self,
        controller_state: ControllerState,
        *,
        action_repeat: int,
        preset: str,
        frame_stack: int,
        stuck_min_speed_kph: float,
        reverse_progress_epsilon: float,
        energy_loss_epsilon: float,
        wrong_way_progress_epsilon: float,
    ) -> BackendStepResult: ...

    def set_controller_state(self, controller_state: ControllerState) -> None: ...

    def render(self) -> np.ndarray: ...

    def observation_spec(self, preset: str) -> ObservationSpec: ...

    def render_display(self, *, preset: str) -> np.ndarray: ...

    def render_observation(self, *, preset: str, frame_stack: int) -> np.ndarray: ...

    def try_read_telemetry(self) -> FZeroXTelemetry | None: ...

    def close(self) -> None: ...
