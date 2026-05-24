# src/rl_fzerox/core/envs/observations/value.py
from __future__ import annotations

from collections.abc import Mapping
from typing import NotRequired, TypeAlias, TypedDict

import numpy as np
from gymnasium import spaces

from fzerox_emulator import FZeroXTelemetry, ObservationSpec, ObservationStackMode
from fzerox_emulator.arrays import ObservationFrame, StateVector
from rl_fzerox.core.domain.observation_components import StateComponentsSettings

from .image import build_image_observation_space
from .state import ObservationMode, state_vector_spec, telemetry_state_vector

ImageObservation: TypeAlias = ObservationFrame


class ImageStateObservation(TypedDict):
    image: ObservationFrame
    state: StateVector
    auxiliary_state_targets: NotRequired[StateVector]


ObservationValue: TypeAlias = ImageObservation | ImageStateObservation

__all__ = [
    "ImageObservation",
    "ImageStateObservation",
    "ObservationValue",
    "build_observation",
    "build_observation_space",
    "observation_image",
    "observation_state",
]


def build_observation(
    *,
    image: ObservationFrame,
    telemetry: FZeroXTelemetry | None,
    mode: ObservationMode,
    action_history: Mapping[str, float] | None = None,
    state_components: StateComponentsSettings | None = None,
    split_lean_history: bool = False,
) -> ObservationValue:
    if mode == "image":
        return image
    if mode == "image_state":
        if state_components is None:
            raise ValueError("image_state observations require state_components")
        return {
            "image": image,
            "state": telemetry_state_vector(
                telemetry,
                state_components=state_components,
                action_history=action_history,
                split_lean_history=split_lean_history,
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
    state_components: StateComponentsSettings | None = None,
    split_lean_history: bool = False,
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
        if state_components is None:
            raise ValueError("image_state observations require state_components")
        spec = state_vector_spec(
            state_components=state_components,
            split_lean_history=split_lean_history,
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
