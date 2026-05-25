# src/fzerox_emulator/base/backend.py
"""Structural emulator backend contract consumed by env and watch code."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from fzerox_emulator.arrays import ObservationFrame, RgbFrame
from fzerox_emulator.base.observations import (
    ObservationImageRecipe,
    ObservationResizeFilter,
    ObservationSpec,
    ObservationStackMode,
)
from fzerox_emulator.base.results import (
    BackendMultiObservationStepResult,
    BackendStepResult,
    FrameStep,
    ResetState,
)
from fzerox_emulator.control import ControllerState, RaceControlState
from fzerox_emulator.control.spin import SpinRequest

if TYPE_CHECKING:
    from fzerox_emulator._native import FZeroXTelemetry


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
        control_state: RaceControlState,
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
        spin_request: SpinRequest = "none",
    ) -> BackendStepResult: ...

    def step_repeat_watch_raw(
        self,
        control_state: RaceControlState,
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
        spin_request: SpinRequest = "none",
    ) -> BackendStepResult: ...

    def step_repeat_multi_observation_raw(
        self,
        control_state: RaceControlState,
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
        spin_request: SpinRequest = "none",
    ) -> BackendMultiObservationStepResult: ...

    def set_controller_state(self, controller_state: ControllerState) -> None: ...

    def load_baseline(self, path: Path) -> None: ...

    def load_baseline_bytes(self, state: bytes, *, source_path: Path | None = None) -> None: ...

    def patch_race_start_setup(
        self,
        *,
        mode: str,
        course_index: int,
        character_index: int,
        engine_setting_raw_value: int,
        total_lap_count: int,
        gp_difficulty_raw_value: int = -1,
    ) -> None: ...

    def patch_machine_settings(
        self,
        *,
        mode: str,
        course_index: int,
        character_index: int,
        engine_setting_raw_value: int,
        total_lap_count: int,
        gp_difficulty_raw_value: int = -1,
    ) -> None: ...

    def patch_time_attack_menu_mode(self) -> None: ...

    def patch_engine_settings(
        self,
        *,
        mode: str,
        engine_setting_raw_value: int,
    ) -> None: ...

    def force_race_reinit(self, *, mode: str) -> None: ...

    def validate_race_start_setup(
        self,
        *,
        mode: str,
        course_index: int,
        character_index: int,
        engine_setting_raw_value: int,
        total_lap_count: int,
        gp_difficulty_raw_value: int = -1,
    ) -> None: ...

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
