# tests/core/envs/helpers.py
from __future__ import annotations

import numpy as np

from fzerox_emulator import (
    BackendStepResult,
    ControllerState,
    FZeroXTelemetry,
    ObservationStackMode,
    StepStatus,
    StepSummary,
    stacked_observation_channels,
)
from fzerox_emulator.arrays import ObservationFrame
from rl_fzerox.core.envs.observations import (
    ObservationValue,
)
from tests.support.fakes import SyntheticBackend
from tests.support.native_objects import make_step_status, make_step_summary, make_telemetry


class ScriptedStepBackend(SyntheticBackend):
    def __init__(
        self,
        results: list[BackendStepResult],
        *,
        reset_telemetry: FZeroXTelemetry | None = None,
    ) -> None:
        super().__init__()
        self._results = list(results)
        self._reset_telemetry = reset_telemetry
        self.last_lean_timer_assist: bool | None = None
        self.last_stuck_step_limit: int | None = None
        self.last_wrong_way_timer_limit: int | None = None

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
        stuck_step_limit: int,
        wrong_way_timer_limit: int | None,
        progress_frontier_stall_limit_frames: int | None,
        progress_frontier_epsilon: float,
        terminate_on_energy_depleted: bool,
        lean_timer_assist: bool = False,
    ) -> BackendStepResult:
        _ = (
            stuck_min_speed_kph,
            energy_loss_epsilon,
            max_episode_steps,
            stuck_step_limit,
            wrong_way_timer_limit,
            progress_frontier_stall_limit_frames,
            progress_frontier_epsilon,
            terminate_on_energy_depleted,
            lean_timer_assist,
            resize_filter,
            minimap_resize_filter,
        )
        self.set_controller_state(controller_state)
        self.last_lean_timer_assist = lean_timer_assist
        self.last_stuck_step_limit = stuck_step_limit
        self.last_wrong_way_timer_limit = wrong_way_timer_limit
        result = self._results.pop(0)
        frames_run = result.summary.frames_run
        self._capture_video_flags.extend([False] * max(frames_run - 1, 0))
        self._capture_video_flags.append(True)
        self._state.frame_index = result.summary.final_frame_index
        self._state.progress = result.summary.max_race_distance
        self._last_frame = self._build_frame()
        expected_channels = stacked_observation_channels(
            3,
            frame_stack=frame_stack,
            stack_mode=stack_mode,
            minimap_layer=minimap_layer,
        )
        if result.observation.shape[2] != expected_channels:
            raise AssertionError("Scripted observation stack does not match frame_stack")
        if preset != "crop_116x164":
            raise AssertionError(f"Unexpected preset {preset!r}")
        return result

    def try_read_telemetry(self) -> FZeroXTelemetry | None:
        return self._reset_telemetry


class CameraSyncBackend(SyntheticBackend):
    def __init__(self, *, camera_setting_raw: int = 2) -> None:
        super().__init__()
        self.camera_setting_raw = camera_setting_raw

    def step_frame(self):
        if self.last_controller_state.right_stick_x > 0.5:
            self.camera_setting_raw = (self.camera_setting_raw + 1) % 4
        return super().step_frame()

    def try_read_telemetry(self) -> FZeroXTelemetry | None:
        return make_telemetry(
            game_mode_raw=1,
            game_mode_name="gp_race",
            in_race_mode=True,
            race_distance=0.0,
            camera_setting_raw=self.camera_setting_raw,
            camera_setting_name=camera_setting_name(self.camera_setting_raw),
        )


def telemetry(
    *,
    race_distance: float,
    state_labels: tuple[str, ...] = ("active",),
    speed_kph: float = 100.0,
    energy: float = 178.0,
    max_energy: float = 178.0,
    boost_timer: int = 0,
    reverse_timer: int = 0,
    lap: int = 1,
    laps_completed: int = 0,
    camera_setting_raw: int = 2,
    camera_setting_name: str = "regular",
) -> FZeroXTelemetry:
    return make_telemetry(
        race_distance=race_distance,
        state_labels=state_labels,
        speed_kph=speed_kph,
        energy=energy,
        max_energy=max_energy,
        boost_timer=boost_timer,
        reverse_timer=reverse_timer,
        lap=lap,
        laps_completed=laps_completed,
        camera_setting_raw=camera_setting_raw,
        camera_setting_name=camera_setting_name,
    )


def camera_setting_name(camera_setting_raw: int) -> str:
    if camera_setting_raw == 0:
        return "overhead"
    if camera_setting_raw == 1:
        return "close_behind"
    if camera_setting_raw == 2:
        return "regular"
    if camera_setting_raw == 3:
        return "wide"
    return "unknown"


def image_obs(observation: ObservationValue) -> ObservationFrame:
    assert isinstance(observation, np.ndarray)
    return observation


def step_summary(
    *,
    max_race_distance: float,
    frames_run: int = 1,
    reverse_active_frames: int = 0,
    low_speed_frames: int = 0,
    energy_loss_total: float = 0.0,
    damage_taken_frames: int = 0,
    consecutive_low_speed_frames: int = 0,
    entered_state_labels: tuple[str, ...] = (),
    final_frame_index: int = 1,
) -> StepSummary:
    return make_step_summary(
        frames_run=frames_run,
        max_race_distance=max_race_distance,
        reverse_active_frames=reverse_active_frames,
        low_speed_frames=low_speed_frames,
        energy_loss_total=energy_loss_total,
        damage_taken_frames=damage_taken_frames,
        consecutive_low_speed_frames=consecutive_low_speed_frames,
        entered_state_labels=entered_state_labels,
        final_frame_index=final_frame_index,
    )


def backend_step_result(
    *,
    telemetry: FZeroXTelemetry,
    summary: StepSummary,
    status: StepStatus | None = None,
) -> BackendStepResult:
    value = np.uint8(summary.final_frame_index % 255)
    observation = np.full((116, 164, 12), value, dtype=np.uint8)
    return BackendStepResult(
        observation=observation,
        summary=summary,
        status=(
            status
            if status is not None
            else make_step_status(
                step_count=summary.final_frame_index,
                termination_reason=telemetry.player.terminal_reason,
            )
        ),
        telemetry=telemetry,
    )
