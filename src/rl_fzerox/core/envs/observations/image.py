# src/rl_fzerox/core/envs/observations/image.py
"""Image observation space and shape helpers.

The emulator produces frame buffers; this module describes how stacking mode
and frame count translate into Gym spaces and tensor shapes. Pixel capture and
state-vector construction live elsewhere.
"""
from __future__ import annotations

import numpy as np
from gymnasium import spaces

from fzerox_emulator import ObservationSpec, ObservationStackMode, stacked_observation_channels

__all__ = [
    "build_image_observation_space",
    "image_observation_shape",
]


def build_image_observation_space(
    observation_spec: ObservationSpec,
    *,
    frame_stack: int,
    stack_mode: ObservationStackMode = "rgb",
    minimap_layer: bool = False,
) -> spaces.Box:
    return spaces.Box(
        low=0,
        high=255,
        shape=image_observation_shape(
            observation_spec,
            frame_stack=frame_stack,
            stack_mode=stack_mode,
            minimap_layer=minimap_layer,
        ),
        dtype=np.uint8,
    )


def image_observation_shape(
    observation_spec: ObservationSpec,
    *,
    frame_stack: int,
    stack_mode: ObservationStackMode = "rgb",
    minimap_layer: bool = False,
) -> tuple[int, int, int]:
    return (
        observation_spec.height,
        observation_spec.width,
        stacked_observation_channels(
            observation_spec.channels,
            frame_stack=frame_stack,
            stack_mode=stack_mode,
            minimap_layer=minimap_layer,
        ),
    )
