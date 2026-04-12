# src/fzerox_emulator/base.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from fzerox_emulator._native import FZeroXTelemetry, StepStatus, StepSummary
from fzerox_emulator.control import ControllerState


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


@dataclass(frozen=True)
class BackendStepResult:
    """Native repeated-step payload consumed by the env engine."""

    observation: np.ndarray
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
        progress_frontier_stall_limit_frames: int | None,
        progress_frontier_epsilon: float,
        terminate_on_energy_depleted: bool,
        shoulder_slide_timer_assist: bool = False,
    ) -> BackendStepResult: ...

    def set_controller_state(self, controller_state: ControllerState) -> None: ...

    def render(self) -> np.ndarray: ...

    def observation_spec(self, preset: str) -> ObservationSpec: ...

    def render_display(self, *, preset: str) -> np.ndarray: ...

    def render_observation(self, *, preset: str, frame_stack: int) -> np.ndarray: ...

    def try_read_telemetry(self) -> FZeroXTelemetry | None: ...

    def close(self) -> None: ...
