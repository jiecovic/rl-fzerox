# src/rl_fzerox/ui/watch/runtime/cnn.py
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol

import numpy as np

from fzerox_emulator.arrays import Float32Array, RgbFrame

if TYPE_CHECKING:
    from rl_fzerox.core.envs.observations import ObservationValue
    from rl_fzerox.core.training.inference import PolicyCnnActivation

_MAX_ACTIVATION_GRID_CHANNELS = 128
CnnActivationNormalizationMode = Literal["channel", "layer_percentile"]
DEFAULT_CNN_ACTIVATION_NORMALIZATION: CnnActivationNormalizationMode = "channel"
_LAYER_PERCENTILE_HIGH = 99.0


class _CnnActivationRunner(Protocol):
    def cnn_activations(
        self,
        observation: ObservationValue,
    ) -> tuple[PolicyCnnActivation, ...]: ...


@dataclass(frozen=True, slots=True)
class CnnActivationLayer:
    """One pre-rendered CNN activation grid for the watch UI."""

    name: str
    image: RgbFrame
    channel_count: int
    spatial_shape: tuple[int, int]
    grid_shape: tuple[int, int]
    rendered_channel_count: int
    normalization: CnnActivationNormalizationMode


@dataclass(frozen=True, slots=True)
class CnnActivationSnapshot:
    """Watch-safe CNN activation payload."""

    layers: tuple[CnnActivationLayer, ...]
    normalization: CnnActivationNormalizationMode = DEFAULT_CNN_ACTIVATION_NORMALIZATION
    error: str | None = None


class CnnActivationSampler:
    """Throttle expensive policy-CNN activation captures for watch mode."""

    def __init__(self, *, refresh_interval_steps: int = 6) -> None:
        self._refresh_interval_steps = max(1, int(refresh_interval_steps))
        self._steps_until_refresh = 0
        self._cached: CnnActivationSnapshot | None = None

    def capture(
        self,
        *,
        enabled: bool,
        policy_runner: _CnnActivationRunner | None,
        observation: ObservationValue,
        normalization: CnnActivationNormalizationMode = DEFAULT_CNN_ACTIVATION_NORMALIZATION,
    ) -> CnnActivationSnapshot | None:
        if not enabled or policy_runner is None:
            self._steps_until_refresh = 0
            return None
        if (
            self._cached is not None
            and self._cached.normalization == normalization
            and self._steps_until_refresh > 0
        ):
            self._steps_until_refresh -= 1
            return self._cached

        self._cached = _capture_policy_cnn_activations(
            policy_runner,
            observation,
            normalization=normalization,
        )
        self._steps_until_refresh = self._refresh_interval_steps - 1
        return self._cached


def _capture_policy_cnn_activations(
    policy_runner: _CnnActivationRunner,
    observation: ObservationValue,
    *,
    normalization: CnnActivationNormalizationMode,
) -> CnnActivationSnapshot:
    try:
        activations = policy_runner.cnn_activations(observation)
    except Exception as exc:
        return CnnActivationSnapshot(layers=(), normalization=normalization, error=str(exc))
    return CnnActivationSnapshot(
        layers=tuple(
            _activation_layer(activation, normalization=normalization) for activation in activations
        ),
        normalization=normalization,
    )


def _activation_layer(
    activation: PolicyCnnActivation,
    *,
    normalization: CnnActivationNormalizationMode,
) -> CnnActivationLayer:
    values = activation.values
    if values.ndim != 3:
        raise ValueError(f"Expected CxHxW activation map, got {values.shape!r}")
    channel_count, height, width = values.shape
    rendered_channel_count = min(channel_count, _MAX_ACTIVATION_GRID_CHANNELS)
    return CnnActivationLayer(
        name=activation.name,
        image=_activation_grid(values, normalization=normalization),
        channel_count=int(channel_count),
        spatial_shape=(int(height), int(width)),
        grid_shape=_activation_grid_shape(rendered_channel_count),
        rendered_channel_count=int(rendered_channel_count),
        normalization=normalization,
    )


def _activation_grid(
    values: Float32Array,
    *,
    max_channels: int = _MAX_ACTIVATION_GRID_CHANNELS,
    normalization: CnnActivationNormalizationMode = DEFAULT_CNN_ACTIVATION_NORMALIZATION,
) -> RgbFrame:
    channel_count, tile_height, tile_width = values.shape
    selected = values[: min(channel_count, max_channels)]
    scale = _activation_scale(selected, normalization=normalization)
    rows, columns = _activation_grid_shape(len(selected))
    grid_height = rows * tile_height
    grid_width = columns * tile_width
    grid = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)

    for index, channel in enumerate(selected):
        row, column = divmod(index, columns)
        y = row * tile_height
        x = column * tile_width
        normalized = _normalized_channel(channel, scale=scale)
        grid[y : y + tile_height, x : x + tile_width] = np.repeat(
            normalized[:, :, None],
            3,
            axis=2,
        )

    return np.ascontiguousarray(grid)


def next_cnn_activation_normalization(
    mode: CnnActivationNormalizationMode,
) -> CnnActivationNormalizationMode:
    if mode == "channel":
        return "layer_percentile"
    return "channel"


def _activation_grid_shape(channel_count: int) -> tuple[int, int]:
    columns = _grid_columns(channel_count)
    rows = max(1, math.ceil(max(1, channel_count) / columns))
    return (rows, columns)


def _grid_columns(channel_count: int) -> int:
    if channel_count <= 0:
        return 1
    return max(1, min(16, math.ceil(math.sqrt(channel_count))))


def _activation_scale(
    values: Float32Array,
    *,
    normalization: CnnActivationNormalizationMode,
) -> tuple[float, float] | None:
    if normalization == "channel":
        return None

    finite = np.nan_to_num(values.astype(np.float32), copy=False)
    if finite.size == 0:
        return (0.0, 0.0)
    high = float(np.percentile(finite, _LAYER_PERCENTILE_HIGH))
    return (0.0, high)


def _normalized_channel(
    channel: Float32Array,
    *,
    scale: tuple[float, float] | None,
) -> np.ndarray:
    finite = np.nan_to_num(channel.astype(np.float32), copy=False)
    if scale is None:
        min_value = float(finite.min(initial=0.0))
        max_value = float(finite.max(initial=0.0))
    else:
        min_value, max_value = scale
    if max_value <= min_value:
        return np.zeros(finite.shape, dtype=np.uint8)
    scaled = (finite - min_value) * (255.0 / (max_value - min_value))
    return np.clip(scaled, 0.0, 255.0).astype(np.uint8)
