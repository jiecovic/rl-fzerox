# src/fzerox_emulator/repeat/requests.py
from __future__ import annotations

from typing import TYPE_CHECKING

from fzerox_emulator.base import ObservationImageRecipe
from fzerox_emulator.control import ControllerState
from fzerox_emulator.repeat.step_options import RepeatStepConfig

if TYPE_CHECKING:
    from collections.abc import Sequence

    from fzerox_emulator._native import (
        ObservationImageRequestDict,
        RepeatMultiObservationStepRequestDict,
        RepeatObservationStepRequestDict,
        RepeatStepRequestDict,
    )


def native_observation_recipe(recipe: ObservationImageRecipe) -> ObservationImageRequestDict:
    payload: ObservationImageRequestDict = {
        "preset": "" if recipe.preset is None else recipe.preset,
        "frame_stack": recipe.frame_stack,
        "stack_mode": recipe.stack_mode,
        "minimap_layer": recipe.minimap_layer,
        "resize_filter": recipe.resize_filter,
        "minimap_resize_filter": recipe.minimap_resize_filter,
    }
    if recipe.height is not None:
        payload["height"] = recipe.height
    if recipe.width is not None:
        payload["width"] = recipe.width
    return payload


def native_repeat_step_request(
    config: RepeatStepConfig,
    controller_state: ControllerState,
) -> RepeatStepRequestDict:
    payload: RepeatStepRequestDict = {
        "action_repeat": config.action_repeat,
        "stuck_min_speed_kph": config.stuck_min_speed_kph,
        "energy_loss_epsilon": config.energy_loss_epsilon,
        "max_episode_steps": config.max_episode_steps,
        "progress_frontier_epsilon": config.progress_frontier_epsilon,
        "terminate_on_energy_depleted": config.terminate_on_energy_depleted,
        "lean_timer_assist": config.lean_timer_assist,
        "joypad_mask": controller_state.joypad_mask,
        "left_stick_x": controller_state.left_stick_x,
        "left_stick_y": controller_state.left_stick_y,
        "right_stick_x": controller_state.right_stick_x,
        "right_stick_y": controller_state.right_stick_y,
    }
    if config.progress_frontier_stall_limit_frames is not None:
        payload["progress_frontier_stall_limit_frames"] = (
            config.progress_frontier_stall_limit_frames
        )
    return payload


def native_repeat_observation_request(
    config: RepeatStepConfig,
    controller_state: ControllerState,
    recipe: ObservationImageRecipe,
) -> RepeatObservationStepRequestDict:
    return {
        "step": native_repeat_step_request(config, controller_state),
        "observation": native_observation_recipe(recipe),
    }


def native_repeat_multi_observation_request(
    config: RepeatStepConfig,
    controller_state: ControllerState,
    recipes: Sequence[ObservationImageRecipe],
) -> RepeatMultiObservationStepRequestDict:
    return {
        "step": native_repeat_step_request(config, controller_state),
        "observations": [native_observation_recipe(recipe) for recipe in recipes],
    }
