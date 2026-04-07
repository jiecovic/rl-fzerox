# src/rl_fzerox/core/envs/observations/resized_rgb.py
from __future__ import annotations

import numpy as np
from gymnasium import spaces

from rl_fzerox.core.config.schema import ObservationConfig
from rl_fzerox.core.emulator.video import display_size


class ResizedObservationAdapter:
    """Aspect-correct and resize raw emulator frames for policy input."""

    def __init__(self, config: ObservationConfig) -> None:
        self._config = config
        channels = 3 if config.rgb else 1
        self._observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(config.height, config.width, channels),
            dtype=np.uint8,
        )

    @property
    def observation_space(self) -> spaces.Box:
        """Return the Gymnasium observation space for this adapter."""

        return self._observation_space

    def transform(self, frame: np.ndarray, *, info: dict[str, object]) -> np.ndarray:
        """Convert one raw emulator frame into an aspect-correct observation."""

        aspect_ratio = _display_aspect_ratio(info)
        display_width, display_height = display_size(frame.shape, aspect_ratio)
        transformed = frame

        if transformed.shape[1] != display_width or transformed.shape[0] != display_height:
            transformed = _resize_frame(
                transformed,
                width=display_width,
                height=display_height,
            )

        if (
            transformed.shape[1] != self._config.width
            or transformed.shape[0] != self._config.height
        ):
            transformed = _resize_frame(
                transformed,
                width=self._config.width,
                height=self._config.height,
            )

        if self._config.rgb:
            return transformed
        return _to_grayscale(transformed)


def _display_aspect_ratio(info: dict[str, object]) -> float:
    value = info.get("display_aspect_ratio")
    if isinstance(value, int | float):
        return float(value)
    return 0.0


def _resize_frame(frame: np.ndarray, *, width: int, height: int) -> np.ndarray:
    if frame.ndim != 3:
        raise ValueError("Expected an HxWxC frame")

    input_height, input_width, _ = frame.shape
    if input_height == height and input_width == width:
        return np.array(frame, copy=True)

    y_index = np.rint(np.linspace(0, input_height - 1, num=height)).astype(np.intp)
    x_index = np.rint(np.linspace(0, input_width - 1, num=width)).astype(np.intp)
    return np.ascontiguousarray(frame[y_index][:, x_index])


def _to_grayscale(frame: np.ndarray) -> np.ndarray:
    rgb = frame.astype(np.float32, copy=False)
    gray = (
        (0.299 * rgb[:, :, 0])
        + (0.587 * rgb[:, :, 1])
        + (0.114 * rgb[:, :, 2])
    )
    return np.rint(gray).astype(np.uint8)[:, :, None]
