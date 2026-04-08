# src/rl_fzerox/core/policy/extractors.py
from __future__ import annotations

from typing import Literal

import torch
from gymnasium import spaces
from stable_baselines3.common.preprocessing import is_image_space_channels_first
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn

ConvSpec = tuple[tuple[int, tuple[int, int], tuple[int, int]], ...]
NATURE_CNN_CONV_SPEC: ConvSpec = (
    (32, (8, 8), (4, 4)),
    (64, (4, 4), (2, 2)),
    (64, (3, 3), (1, 1)),
)
LEGACY_DEEP_CONV_SPEC: ConvSpec = (
    (32, (8, 8), (4, 4)),
    (64, (4, 4), (2, 2)),
    (64, (3, 3), (2, 2)),
    (128, (3, 3), (1, 1)),
)
SUPPORTED_POLICY_GEOMETRIES: dict[tuple[int, int], ConvSpec] = {
    (84, 116): NATURE_CNN_CONV_SPEC,
    (92, 124): NATURE_CNN_CONV_SPEC,
    (116, 164): LEGACY_DEEP_CONV_SPEC,
}


class FZeroXObservationCnnExtractor(BaseFeaturesExtractor):
    """CNN extractor for the supported aspect-corrected F-Zero X observation presets."""

    def __init__(
        self,
        observation_space: spaces.Box,
        features_dim: int | Literal["auto"] = 512,
    ) -> None:
        if len(observation_space.shape) != 3:
            raise ValueError(
                f"{type(self).__name__} expects a 3D image observation space, "
                f"got {observation_space.shape!r}"
            )

        if is_image_space_channels_first(observation_space):
            channels, height, width = observation_space.shape
        else:
            height, width, channels = observation_space.shape

        self._height = int(height)
        self._width = int(width)
        self._channels = int(channels)
        geometry = (self._height, self._width)
        try:
            self._conv_spec = SUPPORTED_POLICY_GEOMETRIES[geometry]
        except KeyError as error:
            supported = ", ".join(
                f"{height}x{width}" for height, width in SUPPORTED_POLICY_GEOMETRIES
            )
            raise ValueError(
                f"Unsupported observation geometry {self._height}x{self._width} for "
                f"{type(self).__name__}; supported presets: {supported}"
            ) from error

        cnn = nn.Sequential(
            *self._build_conv_layers(
                input_channels=self._channels,
                conv_spec=self._conv_spec,
            ),
            nn.Flatten(),
        )

        with torch.no_grad():
            sample = torch.zeros(1, self._channels, self._height, self._width)
            n_flatten = int(cnn(sample).shape[1])

        resolved_features_dim = n_flatten if features_dim == "auto" else int(features_dim)
        super().__init__(observation_space, resolved_features_dim)
        self._flatten_dim = n_flatten
        self._cnn = cnn
        if features_dim == "auto":
            self._linear: nn.Module = nn.Identity()
        else:
            self._linear = nn.Sequential(
                nn.Linear(n_flatten, resolved_features_dim),
                nn.ReLU(),
            )

    def _build_conv_layers(
        self,
        *,
        input_channels: int,
        conv_spec: ConvSpec,
    ) -> list[nn.Module]:
        layers: list[nn.Module] = []
        in_channels = input_channels
        for out_channels, kernel_size, stride in conv_spec:
            layers.append(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                )
            )
            layers.append(nn.ReLU())
            in_channels = out_channels
        return layers

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Convert either NHWC or NCHW observations into shared PPO features."""

        if observations.ndim != 4:
            raise ValueError(f"Expected a 4D observation tensor, got {tuple(observations.shape)!r}")

        if observations.shape[1:] == (self._channels, self._height, self._width):
            channels_first = observations
        elif observations.shape[1:] == (self._height, self._width, self._channels):
            channels_first = observations.permute(0, 3, 1, 2).contiguous()
        else:
            raise ValueError(
                f"Unexpected observation tensor shape for {type(self).__name__}: "
                f"{tuple(observations.shape)!r}"
            )

        return self._linear(self._cnn(channels_first))
