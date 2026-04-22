# src/rl_fzerox/core/envs/observations.py
from __future__ import annotations

from collections.abc import Collection, Mapping
from typing import TypeAlias, TypedDict

import numpy as np
from gymnasium import spaces

from fzerox_emulator import FZeroXTelemetry, ObservationSpec, ObservationStackMode
from fzerox_emulator.arrays import ObservationFrame, StateVector
from rl_fzerox.core.envs.observation_image import build_image_observation_space
from rl_fzerox.core.envs.observation_state import (
    OBSERVATION_STATE_DEFAULTS,
    ActionHistoryControl,
    ObservationCourseContext,
    ObservationGroundEffectContext,
    ObservationMode,
    ObservationStateProfile,
    StateComponentsSettings,
    StateFeature,
    StateVectorSpec,
    action_history_feature_names,
    action_history_settings_for_observation,
    state_feature_count,
    state_feature_names,
    state_vector_spec,
    telemetry_state_vector,
)

ImageObservation: TypeAlias = ObservationFrame


class ImageStateObservation(TypedDict):
    image: ObservationFrame
    state: StateVector


ObservationValue: TypeAlias = ImageObservation | ImageStateObservation

__all__ = [
    "ActionHistoryControl",
    "ImageObservation",
    "ImageStateObservation",
    "OBSERVATION_STATE_DEFAULTS",
    "ObservationCourseContext",
    "ObservationGroundEffectContext",
    "ObservationMode",
    "ObservationStackMode",
    "ObservationStateProfile",
    "ObservationValue",
    "StateComponentsSettings",
    "StateFeature",
    "StateVectorSpec",
    "action_history_feature_names",
    "action_history_settings_for_observation",
    "build_image_observation_space",
    "build_observation",
    "build_observation_space",
    "observation_image",
    "observation_state",
    "state_feature_count",
    "state_feature_names",
    "state_vector_spec",
    "telemetry_state_vector",
]


def build_observation(
    *,
    image: ObservationFrame,
    telemetry: FZeroXTelemetry | None,
    mode: ObservationMode,
    state_profile: ObservationStateProfile = OBSERVATION_STATE_DEFAULTS.state_profile,
    course_context: ObservationCourseContext = OBSERVATION_STATE_DEFAULTS.course_context,
    ground_effect_context: ObservationGroundEffectContext = (
        OBSERVATION_STATE_DEFAULTS.ground_effect_context
    ),
    left_lean_held: float = 0.0,
    right_lean_held: float = 0.0,
    left_press_age_norm: float = 1.0,
    right_press_age_norm: float = 1.0,
    recent_boost_pressure: float = 0.0,
    steer_left_held: float = 0.0,
    steer_right_held: float = 0.0,
    recent_steer_pressure: float = 0.0,
    action_history_len: int | None = OBSERVATION_STATE_DEFAULTS.action_history_len,
    action_history_controls: tuple[
        ActionHistoryControl, ...
    ] = OBSERVATION_STATE_DEFAULTS.action_history_controls,
    action_history: Mapping[str, float] | None = None,
    state_components: StateComponentsSettings | None = None,
    zeroed_state_components: Collection[str] = (),
) -> ObservationValue:
    if mode == "image":
        return image
    if mode == "image_state":
        return {
            "image": image,
            "state": telemetry_state_vector(
                telemetry,
                state_profile=state_profile,
                course_context=course_context,
                ground_effect_context=ground_effect_context,
                left_lean_held=left_lean_held,
                right_lean_held=right_lean_held,
                left_press_age_norm=left_press_age_norm,
                right_press_age_norm=right_press_age_norm,
                recent_boost_pressure=recent_boost_pressure,
                steer_left_held=steer_left_held,
                steer_right_held=steer_right_held,
                recent_steer_pressure=recent_steer_pressure,
                action_history_len=action_history_len,
                action_history_controls=action_history_controls,
                action_history=action_history,
                state_components=state_components,
                zeroed_state_components=zeroed_state_components,
            ),
        }
    raise ValueError(f"Unsupported observation mode: {mode!r}")


def build_observation_space(
    observation_spec: ObservationSpec,
    *,
    frame_stack: int,
    stack_mode: ObservationStackMode = "rgb",
    minimap_layer: bool = False,
    mode: ObservationMode,
    state_profile: ObservationStateProfile = OBSERVATION_STATE_DEFAULTS.state_profile,
    course_context: ObservationCourseContext = OBSERVATION_STATE_DEFAULTS.course_context,
    ground_effect_context: ObservationGroundEffectContext = (
        OBSERVATION_STATE_DEFAULTS.ground_effect_context
    ),
    action_history_len: int | None = OBSERVATION_STATE_DEFAULTS.action_history_len,
    action_history_controls: tuple[
        ActionHistoryControl, ...
    ] = OBSERVATION_STATE_DEFAULTS.action_history_controls,
    state_components: StateComponentsSettings | None = None,
) -> spaces.Box | spaces.Dict:
    image_space = build_image_observation_space(
        observation_spec,
        frame_stack=frame_stack,
        stack_mode=stack_mode,
        minimap_layer=minimap_layer,
    )
    if mode == "image":
        return image_space
    if mode == "image_state":
        spec = state_vector_spec(
            state_profile,
            course_context=course_context,
            ground_effect_context=ground_effect_context,
            action_history_len=action_history_len,
            action_history_controls=action_history_controls,
            state_components=state_components,
        )
        return spaces.Dict(
            {
                "image": image_space,
                "state": spaces.Box(
                    low=spec.low_array(),
                    high=spec.high_array(),
                    shape=(spec.count,),
                    dtype=np.float32,
                ),
            }
        )
    raise ValueError(f"Unsupported observation mode: {mode!r}")


def observation_image(observation: ObservationValue) -> ObservationFrame:
    if isinstance(observation, dict):
        image = observation.get("image")
        if not isinstance(image, np.ndarray):
            raise ValueError("Dict observation is missing ndarray key 'image'")
        return image
    return observation


def observation_state(observation: ObservationValue) -> StateVector | None:
    if not isinstance(observation, dict):
        return None
    state = observation.get("state")
    if state is None:
        return None
    if not isinstance(state, np.ndarray):
        raise ValueError("Dict observation key 'state' must be an ndarray")
    return state
