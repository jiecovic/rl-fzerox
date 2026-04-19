# src/rl_fzerox/core/envs/observation_image.py
from __future__ import annotations

import numpy as np
from gymnasium import spaces

from fzerox_emulator import ObservationSpec, ObservationStackMode, stacked_observation_channels


def build_image_observation_space(
    observation_spec: ObservationSpec,
    *,
    frame_stack: int,
    stack_mode: ObservationStackMode = "rgb",
) -> spaces.Box:
    return spaces.Box(
        low=0,
        high=255,
        shape=image_observation_shape(
            observation_spec,
            frame_stack=frame_stack,
            stack_mode=stack_mode,
        ),
        dtype=np.uint8,
    )


def image_observation_shape(
    observation_spec: ObservationSpec,
    *,
    frame_stack: int,
    stack_mode: ObservationStackMode = "rgb",
) -> tuple[int, int, int]:
    return (
        observation_spec.height,
        observation_spec.width,
        stacked_observation_channels(
            observation_spec.channels,
            frame_stack=frame_stack,
            stack_mode=stack_mode,
        ),
    )
