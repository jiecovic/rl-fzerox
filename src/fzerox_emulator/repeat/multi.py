# src/fzerox_emulator/repeat/multi.py
from __future__ import annotations

from collections.abc import Sequence

from fzerox_emulator._native import Emulator as NativeEmulator
from fzerox_emulator.base import BackendMultiObservationStepResult, ObservationImageRecipe
from fzerox_emulator.control import ControllerState
from fzerox_emulator.repeat.requests import native_observation_recipe
from fzerox_emulator.repeat.step_options import RepeatStepConfig


def run_repeat_multi_observation_step(
    native: NativeEmulator,
    controller_state: ControllerState,
    *,
    config: RepeatStepConfig,
    recipes: Sequence[ObservationImageRecipe],
) -> BackendMultiObservationStepResult:
    if len(recipes) == 0:
        raise ValueError("At least one observation recipe is required")

    raw_recipes = [native_observation_recipe(recipe) for recipe in recipes]
    observations, summary, status, telemetry = native.step_repeat_multi_observation_raw(
        action_repeat=config.action_repeat,
        observation_requests=raw_recipes,
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
    return BackendMultiObservationStepResult(
        observations=tuple(observations),
        summary=summary,
        status=status,
        telemetry=telemetry,
    )
