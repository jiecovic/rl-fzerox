# src/rl_fzerox/core/policy/extractors.py
from __future__ import annotations

import torch
from gymnasium import spaces
from stable_baselines3.common.preprocessing import is_image_space_channels_first
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn


class FZeroXCnnExtractor(BaseFeaturesExtractor):
    """Custom CNN for stacked `160x120` F-Zero X RGB observations."""

    def __init__(self, observation_space: spaces.Box, features_dim: int = 512) -> None:
        if len(observation_space.shape) != 3:
            raise ValueError(
                "FZeroXCnnExtractor expects a 3D image observation space, "
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
        self._cnn = nn.Sequential(
            nn.Conv2d(self._channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            sample = torch.zeros(1, self._channels, self._height, self._width)
            n_flatten = int(self._cnn(sample).shape[1])

        self._linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )

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
                "Unexpected observation tensor shape for FZeroXCnnExtractor: "
                f"{tuple(observations.shape)!r}"
            )

        return self._linear(self._cnn(channels_first))
