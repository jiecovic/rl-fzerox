# src/rl_fzerox/core/policy/extractors.py
from __future__ import annotations

import torch
from gymnasium import spaces
from stable_baselines3.common.preprocessing import is_image_space_channels_first
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn


class _BaseFZeroXCnnExtractor(BaseFeaturesExtractor):
    """Shared CNN extractor base for stacked `160x120` F-Zero X RGB observations."""

    def __init__(
        self,
        observation_space: spaces.Box,
        *,
        conv_spec: tuple[tuple[int, int, int], ...],
        features_dim: int = 512,
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

        super().__init__(observation_space, features_dim)
        self._height = int(height)
        self._width = int(width)
        self._channels = int(channels)
        self._cnn = nn.Sequential(*self._build_conv_layers(conv_spec), nn.Flatten())

        with torch.no_grad():
            sample = torch.zeros(1, self._channels, self._height, self._width)
            n_flatten = int(self._cnn(sample).shape[1])

        self._linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )

    def _build_conv_layers(
        self,
        conv_spec: tuple[tuple[int, int, int], ...],
    ) -> list[nn.Module]:
        layers: list[nn.Module] = []
        in_channels = self._channels
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
            raise ValueError(
                f"Expected a 4D observation tensor, got {tuple(observations.shape)!r}"
            )

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


class FZeroXCnnExtractor(_BaseFZeroXCnnExtractor):
    """Original CNN kept stable for existing checkpoints."""

    def __init__(self, observation_space: spaces.Box, features_dim: int = 512) -> None:
        super().__init__(
            observation_space,
            conv_spec=(
                (32, 8, 4),
                (64, 4, 2),
                (64, 3, 2),
                (128, 3, 1),
            ),
            features_dim=features_dim,
        )


class FZeroXCnnWideExtractor(_BaseFZeroXCnnExtractor):
    """Wider CNN variant for stacked RGB frames with higher early capacity."""

    def __init__(self, observation_space: spaces.Box, features_dim: int = 512) -> None:
        super().__init__(
            observation_space,
            conv_spec=(
                (64, 8, 4),
                (64, 4, 2),
                (128, 3, 2),
                (128, 3, 1),
            ),
            features_dim=features_dim,
        )
