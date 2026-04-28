# src/fzerox_emulator/frames.py
from __future__ import annotations

import numpy as np

from fzerox_emulator.arrays import ObservationFrame, RgbFrame
from fzerox_emulator.base import ObservationSpec, ObservationStackMode, stacked_observation_channels


def expected_observation_shape(
    spec: ObservationSpec,
    *,
    frame_stack: int,
    stack_mode: ObservationStackMode,
    minimap_layer: bool,
) -> tuple[int, int, int]:
    stacked_channels = stacked_observation_channels(
        spec.channels,
        frame_stack=frame_stack,
        stack_mode=stack_mode,
        minimap_layer=minimap_layer,
    )
    return spec.height, spec.width, stacked_channels


def validated_observation_frame(
    frame: object,
    *,
    expected_shape: tuple[int, int, int],
    source_label: str,
) -> ObservationFrame:
    observation = np.asarray(frame, dtype=np.uint8)
    expected_size = expected_shape[0] * expected_shape[1] * expected_shape[2]
    if observation.size != expected_size:
        raise RuntimeError(
            f"Unexpected observation size from {source_label}: "
            f"expected {expected_size} bytes, got {observation.size}"
        )
    if tuple(int(value) for value in observation.shape) != expected_shape:
        raise RuntimeError(
            f"Unexpected observation frame shape from {source_label}: "
            f"expected {expected_shape!r}, got {tuple(observation.shape)!r}"
        )
    return np.ascontiguousarray(observation)


def validated_raw_rgb_frame(frame_bytes: bytes, frame_shape: tuple[int, int, int]) -> RgbFrame:
    frame_height, frame_width, channels = frame_shape
    frame = np.frombuffer(frame_bytes, dtype=np.uint8)
    expected_size = frame_height * frame_width * channels
    if frame.size != expected_size:
        raise RuntimeError(
            "Unexpected frame size from native emulator: "
            f"expected {expected_size} bytes, got {frame.size}"
        )
    return frame.reshape((frame_height, frame_width, channels))


def validated_display_frame(frame: object, *, expected_shape: tuple[int, int, int]) -> RgbFrame:
    display_frame = np.asarray(frame, dtype=np.uint8)
    if tuple(int(value) for value in display_frame.shape) != expected_shape:
        raise RuntimeError(
            "Unexpected display frame shape from native emulator: "
            f"expected {expected_shape!r}, got {tuple(display_frame.shape)!r}"
        )
    return np.ascontiguousarray(display_frame)
