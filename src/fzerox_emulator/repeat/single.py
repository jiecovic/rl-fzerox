# src/fzerox_emulator/repeat/single.py
from __future__ import annotations

from fzerox_emulator._native import Emulator as NativeEmulator
from fzerox_emulator.base import BackendStepResult, ObservationImageRecipe
from fzerox_emulator.control import ControllerState
from fzerox_emulator.repeat.step_options import RepeatStepConfig


def run_repeat_step(
    native: NativeEmulator,
    controller_state: ControllerState,
    *,
    config: RepeatStepConfig,
    recipe: ObservationImageRecipe,
) -> BackendStepResult:
    observation, summary, status, telemetry = native.step_repeat_raw(
        action_repeat=config.action_repeat,
        preset="" if recipe.preset is None else recipe.preset,
        height=recipe.height,
        width=recipe.width,
        frame_stack=recipe.frame_stack,
        stack_mode=recipe.stack_mode,
        minimap_layer=recipe.minimap_layer,
        resize_filter=recipe.resize_filter,
        minimap_resize_filter=recipe.minimap_resize_filter,
        stuck_min_speed_kph=config.stuck_min_speed_kph,
        energy_loss_epsilon=config.energy_loss_epsilon,
        max_episode_steps=config.max_episode_steps,
        progress_frontier_stall_limit_frames=config.progress_frontier_stall_limit_frames,
        progress_frontier_epsilon=config.progress_frontier_epsilon,
        terminate_on_energy_depleted=config.terminate_on_energy_depleted,
        lean_timer_assist=config.lean_timer_assist,
        joypad_mask=controller_state.joypad_mask,
        left_stick_x=controller_state.left_stick_x,
        left_stick_y=controller_state.left_stick_y,
        right_stick_x=controller_state.right_stick_x,
        right_stick_y=controller_state.right_stick_y,
    )
    return BackendStepResult(
        observation=observation,
        summary=summary,
        status=status,
        telemetry=telemetry,
    )
