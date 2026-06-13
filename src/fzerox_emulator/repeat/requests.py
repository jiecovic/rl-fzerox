# src/fzerox_emulator/repeat/requests.py
"""Builders for the dictionary payloads passed into native repeated-step APIs."""

from __future__ import annotations

from typing import TYPE_CHECKING

from fzerox_emulator.base.observations import ObservationImageRecipe
from fzerox_emulator.control import RaceControlState
from fzerox_emulator.repeat.step_options import RepeatStepConfig

if TYPE_CHECKING:
    from collections.abc import Sequence

    from fzerox_emulator.boundary import (
        ObservationImageRequestDict,
        RepeatMultiObservationStepRequestDict,
        RepeatObservationStepRequestDict,
        RepeatStepRequestDict,
    )


def native_observation_recipe(recipe: ObservationImageRecipe) -> ObservationImageRequestDict:
    """Translate one Python observation recipe into its native request payload."""

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
    control_state: RaceControlState,
) -> RepeatStepRequestDict:
    """Translate repeat-step config and semantic race controls into a native payload."""

    payload: RepeatStepRequestDict = {
        "action_repeat": config.action_repeat,
        "stuck_min_speed_kph": config.stuck_min_speed_kph,
        "energy_loss_epsilon": config.energy_loss_epsilon,
        "max_episode_steps": config.max_episode_steps,
        "progress_frontier_epsilon": config.progress_frontier_epsilon,
        "terminate_on_energy_depleted": config.terminate_on_energy_depleted,
        "lean_timer_assist": config.lean_timer_assist,
        "spin_request": config.spin_request,
        "spin_cooldown_frames": config.spin_cooldown_frames,
        "gas": control_state.gas,
        "air_brake": control_state.air_brake,
        "boost": control_state.boost,
        "lean_left": control_state.lean_left,
        "lean_right": control_state.lean_right,
        "stick_x": control_state.stick_x,
        "pitch": control_state.pitch,
    }
    if config.progress_frontier_stall_limit_frames is not None:
        payload["progress_frontier_stall_limit_frames"] = (
            config.progress_frontier_stall_limit_frames
        )
    return payload


def native_repeat_observation_request(
    config: RepeatStepConfig,
    control_state: RaceControlState,
    recipe: ObservationImageRecipe,
    *,
    capture_audio: bool = False,
) -> RepeatObservationStepRequestDict:
    """Build the native request for one repeated step with one observation view."""

    payload: RepeatObservationStepRequestDict = {
        "step": native_repeat_step_request(config, control_state),
        "observation": native_observation_recipe(recipe),
    }
    if capture_audio:
        payload["capture_audio"] = True
    return payload


def native_repeat_multi_observation_request(
    config: RepeatStepConfig,
    control_state: RaceControlState,
    recipes: Sequence[ObservationImageRecipe],
) -> RepeatMultiObservationStepRequestDict:
    """Build the native request for one repeated step with multiple observations."""

    return {
        "step": native_repeat_step_request(config, control_state),
        "observations": [native_observation_recipe(recipe) for recipe in recipes],
    }
