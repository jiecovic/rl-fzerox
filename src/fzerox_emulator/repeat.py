# src/fzerox_emulator/repeat.py
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from fzerox_emulator._native import Emulator as NativeEmulator
from fzerox_emulator.base import (
    BackendMultiObservationStepResult,
    BackendStepResult,
    ObservationImageRecipe,
    ObservationSpec,
)
from fzerox_emulator.control import ControllerState
from fzerox_emulator.frames import (
    expected_observation_shape,
    validated_display_frame,
    validated_observation_frame,
)

if TYPE_CHECKING:
    from fzerox_emulator._native import ObservationImageRequestDict


@dataclass(frozen=True, slots=True)
class RepeatStepConfig:
    action_repeat: int
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
    recipe: ObservationImageRecipe,
    spec: ObservationSpec,
) -> BackendStepResult:
    state = controller_state.clamped()
    preset, height, width = recipe.normalized_resolution()
    observation, summary, status, telemetry = native.step_repeat_raw(
        action_repeat=config.action_repeat,
        preset="" if preset is None else preset,
        height=height,
        width=width,
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
        joypad_mask=state.joypad_mask,
        left_stick_x=state.left_stick_x,
        left_stick_y=state.left_stick_y,
        right_stick_x=state.right_stick_x,
        right_stick_y=state.right_stick_y,
    )
    expected_shape = expected_observation_shape(
        spec,
        frame_stack=recipe.frame_stack,
        stack_mode=recipe.stack_mode,
        minimap_layer=recipe.minimap_layer,
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
    recipe: ObservationImageRecipe,
    spec: ObservationSpec,
) -> BackendStepResult:
    state = controller_state.clamped()
    preset, height, width = recipe.normalized_resolution()
    observation, display_frames, summary, status, telemetry = native.step_repeat_watch_raw(
        action_repeat=config.action_repeat,
        preset="" if preset is None else preset,
        height=height,
        width=width,
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
        joypad_mask=state.joypad_mask,
        left_stick_x=state.left_stick_x,
        left_stick_y=state.left_stick_y,
        right_stick_x=state.right_stick_x,
        right_stick_y=state.right_stick_y,
    )
    expected_shape = expected_observation_shape(
        spec,
        frame_stack=recipe.frame_stack,
        stack_mode=recipe.stack_mode,
        minimap_layer=recipe.minimap_layer,
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


def run_repeat_multi_observation_step(
    native: NativeEmulator,
    controller_state: ControllerState,
    *,
    config: RepeatStepConfig,
    recipes: Sequence[ObservationImageRecipe],
    specs: Sequence[ObservationSpec],
) -> BackendMultiObservationStepResult:
    if len(recipes) == 0:
        raise ValueError("At least one observation recipe is required")
    if len(recipes) != len(specs):
        raise ValueError("recipes and specs must have the same length")

    state = controller_state.clamped()
    raw_recipes = [_native_observation_recipe(recipe) for recipe in recipes]
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
        joypad_mask=state.joypad_mask,
        left_stick_x=state.left_stick_x,
        left_stick_y=state.left_stick_y,
        right_stick_x=state.right_stick_x,
        right_stick_y=state.right_stick_y,
    )
    if len(observations) != len(recipes):
        raise RuntimeError(
            "Unexpected observation count from native repeated step: "
            f"expected {len(recipes)}, got {len(observations)}"
        )

    validated_observations = tuple(
        validated_observation_frame(
            observation,
            expected_shape=expected_observation_shape(
                spec,
                frame_stack=recipe.frame_stack,
                stack_mode=recipe.stack_mode,
                minimap_layer=recipe.minimap_layer,
            ),
            source_label="native repeated step batch",
        )
        for observation, recipe, spec in zip(observations, recipes, specs, strict=True)
    )
    return BackendMultiObservationStepResult(
        observations=validated_observations,
        summary=summary,
        status=status,
        telemetry=telemetry,
    )


def _native_observation_recipe(recipe: ObservationImageRecipe) -> ObservationImageRequestDict:
    preset, height, width = recipe.normalized_resolution()
    payload: ObservationImageRequestDict = {
        "preset": "" if preset is None else preset,
        "frame_stack": recipe.frame_stack,
        "stack_mode": recipe.stack_mode,
        "minimap_layer": recipe.minimap_layer,
        "resize_filter": recipe.resize_filter,
        "minimap_resize_filter": recipe.minimap_resize_filter,
    }
    if height is not None:
        payload["height"] = height
    if width is not None:
        payload["width"] = width
    return payload
