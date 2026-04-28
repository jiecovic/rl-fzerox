# src/fzerox_emulator/repeat.py
from __future__ import annotations

from dataclasses import dataclass

from fzerox_emulator._native import Emulator as NativeEmulator
from fzerox_emulator.base import (
    BackendStepResult,
    ObservationResizeFilter,
    ObservationSpec,
    ObservationStackMode,
)
from fzerox_emulator.control import ControllerState
from fzerox_emulator.frames import (
    expected_observation_shape,
    validated_display_frame,
    validated_observation_frame,
)


@dataclass(frozen=True, slots=True)
class RepeatStepConfig:
    action_repeat: int
    preset: str
    frame_stack: int
    stack_mode: ObservationStackMode
    minimap_layer: bool
    resize_filter: ObservationResizeFilter
    minimap_resize_filter: ObservationResizeFilter
    stuck_min_speed_kph: float
    energy_loss_epsilon: float
    max_episode_steps: int
    progress_frontier_stall_limit_frames: int | None
    progress_frontier_epsilon: float
    terminate_on_energy_depleted: bool
    lean_timer_assist: bool


def run_repeat_step(
    native: NativeEmulator,
    controller_state: ControllerState,
    *,
    config: RepeatStepConfig,
    spec: ObservationSpec,
) -> BackendStepResult:
    state = controller_state.clamped()
    observation, summary, status, telemetry = native.step_repeat_raw(
        action_repeat=config.action_repeat,
        preset=config.preset,
        frame_stack=config.frame_stack,
        stack_mode=config.stack_mode,
        minimap_layer=config.minimap_layer,
        resize_filter=config.resize_filter,
        minimap_resize_filter=config.minimap_resize_filter,
        stuck_min_speed_kph=config.stuck_min_speed_kph,
        energy_loss_epsilon=config.energy_loss_epsilon,
        max_episode_steps=config.max_episode_steps,
        progress_frontier_stall_limit_frames=config.progress_frontier_stall_limit_frames,
        progress_frontier_epsilon=config.progress_frontier_epsilon,
        terminate_on_energy_depleted=config.terminate_on_energy_depleted,
        lean_timer_assist=config.lean_timer_assist,
        joypad_mask=state.joypad_mask,
        left_stick_x=state.left_stick_x,
        left_stick_y=state.left_stick_y,
        right_stick_x=state.right_stick_x,
        right_stick_y=state.right_stick_y,
    )
    expected_shape = expected_observation_shape(
        spec,
        frame_stack=config.frame_stack,
        stack_mode=config.stack_mode,
        minimap_layer=config.minimap_layer,
    )
    frame = validated_observation_frame(
        observation,
        expected_shape=expected_shape,
        source_label="native repeated step",
    )
    return BackendStepResult(
        observation=frame,
        summary=summary,
        status=status,
        telemetry=telemetry,
    )


def run_repeat_watch_step(
    native: NativeEmulator,
    controller_state: ControllerState,
    *,
    config: RepeatStepConfig,
    spec: ObservationSpec,
) -> BackendStepResult:
    state = controller_state.clamped()
    observation, display_frames, summary, status, telemetry = native.step_repeat_watch_raw(
        action_repeat=config.action_repeat,
        preset=config.preset,
        frame_stack=config.frame_stack,
        stack_mode=config.stack_mode,
        minimap_layer=config.minimap_layer,
        resize_filter=config.resize_filter,
        minimap_resize_filter=config.minimap_resize_filter,
        stuck_min_speed_kph=config.stuck_min_speed_kph,
        energy_loss_epsilon=config.energy_loss_epsilon,
        max_episode_steps=config.max_episode_steps,
        progress_frontier_stall_limit_frames=config.progress_frontier_stall_limit_frames,
        progress_frontier_epsilon=config.progress_frontier_epsilon,
        terminate_on_energy_depleted=config.terminate_on_energy_depleted,
        lean_timer_assist=config.lean_timer_assist,
        joypad_mask=state.joypad_mask,
        left_stick_x=state.left_stick_x,
        left_stick_y=state.left_stick_y,
        right_stick_x=state.right_stick_x,
        right_stick_y=state.right_stick_y,
    )
    expected_shape = expected_observation_shape(
        spec,
        frame_stack=config.frame_stack,
        stack_mode=config.stack_mode,
        minimap_layer=config.minimap_layer,
    )
    frame = validated_observation_frame(
        observation,
        expected_shape=expected_shape,
        source_label="native watch step",
    )
    expected_display_shape = (spec.display_height, spec.display_width, 3)
    validated_display_frames = tuple(
        validated_display_frame(display_frame, expected_shape=expected_display_shape)
        for display_frame in display_frames
    )
    if len(validated_display_frames) != config.action_repeat:
        raise RuntimeError(
            "Unexpected display frame count from native watch step: "
            f"expected {config.action_repeat}, got {len(validated_display_frames)}"
        )
    return BackendStepResult(
        observation=frame,
        summary=summary,
        status=status,
        telemetry=telemetry,
        display_frames=validated_display_frames,
    )
