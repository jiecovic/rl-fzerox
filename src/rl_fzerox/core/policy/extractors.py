# src/rl_fzerox/core/policy/extractors.py
from __future__ import annotations

from typing import Literal

import torch
from gymnasium import spaces
from stable_baselines3.common.preprocessing import is_image_space_channels_first
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn

ConvSpec = tuple[tuple[int, tuple[int, int], tuple[int, int]], ...]
ConvProfile = Literal["auto", "nature", "compact_deep", "compact_bottleneck"]
NATURE_CNN_CONV_SPEC: ConvSpec = (
    (32, (8, 8), (4, 4)),
    (64, (4, 4), (2, 2)),
    (64, (3, 3), (1, 1)),
)
LEGACY_DEEP_CONV_SPEC: ConvSpec = (
    (64, (8, 8), (4, 4)),
    (64, (4, 4), (2, 2)),
    (128, (3, 3), (2, 2)),
    (128, (3, 3), (1, 1)),
)
COMPACT_DEEP_CONV_SPEC: ConvSpec = (
    (64, (8, 8), (2, 2)),
    (64, (4, 4), (2, 2)),
    (128, (4, 4), (2, 2)),
    (128, (4, 4), (2, 2)),
)
COMPACT_BOTTLENECK_CONV_SPEC: ConvSpec = (
    (64, (8, 8), (2, 2)),
    (64, (8, 8), (2, 2)),
    (64, (4, 4), (2, 2)),
    (128, (3, 3), (2, 2)),
    (256, (4, 4), (1, 1)),
)
SUPPORTED_POLICY_GEOMETRIES: dict[tuple[int, int], ConvSpec] = {
    (84, 116): NATURE_CNN_CONV_SPEC,
    (92, 124): NATURE_CNN_CONV_SPEC,
    (116, 164): LEGACY_DEEP_CONV_SPEC,
    (98, 130): COMPACT_DEEP_CONV_SPEC,
    (66, 82): COMPACT_DEEP_CONV_SPEC,
}


class FZeroXObservationCnnExtractor(BaseFeaturesExtractor):
    """CNN extractor for the supported aspect-corrected F-Zero X observation presets."""

    def __init__(
        self,
        observation_space: spaces.Box,
        features_dim: int | Literal["auto"] = 512,
        conv_profile: ConvProfile = "auto",
        layer_norm: bool = False,
    ) -> None:
        image_geometry = _resolve_supported_image_geometry(
            observation_space,
            extractor_name=type(self).__name__,
            conv_profile=conv_profile,
        )
        self._height = image_geometry.height
        self._width = image_geometry.width
        self._channels = image_geometry.channels
        self._conv_spec = image_geometry.conv_spec

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
        self._layer_norm: nn.Module = (
            nn.LayerNorm(resolved_features_dim) if layer_norm else nn.Identity()
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

        return self._layer_norm(self._linear(self._cnn(channels_first)))


class FZeroXImageStateExtractor(BaseFeaturesExtractor):
    """CNN image features plus a scalar-state branch for Dict observations."""

    def __init__(
        self,
        observation_space: spaces.Dict,
        features_dim: int | Literal["auto"] = 512,
        state_features_dim: int = 64,
        fusion_features_dim: int | None = None,
        conv_profile: ConvProfile = "auto",
        layer_norm: bool = False,
    ) -> None:
        if not isinstance(observation_space, spaces.Dict):
            raise ValueError(
                f"{type(self).__name__} expects a Dict observation space, "
                f"got {type(observation_space).__name__}"
            )

        image_space = observation_space.spaces.get("image")
        state_space = observation_space.spaces.get("state")
        if not isinstance(image_space, spaces.Box):
            raise ValueError(f"{type(self).__name__} requires Box key 'image'")
        if not isinstance(state_space, spaces.Box) or len(state_space.shape) != 1:
            raise ValueError(f"{type(self).__name__} requires 1D Box key 'state'")

        image_features_dim = (
            _image_flatten_dim(image_space, conv_profile=conv_profile)
            if features_dim == "auto"
            else int(features_dim)
        )
        resolved_state_features_dim = int(state_features_dim)
        combined_features_dim = image_features_dim + resolved_state_features_dim
        resolved_fusion_features_dim = (
            None if fusion_features_dim is None else int(fusion_features_dim)
        )
        output_features_dim = (
            combined_features_dim
            if resolved_fusion_features_dim is None
            else resolved_fusion_features_dim
        )
        super().__init__(observation_space, output_features_dim)

        self._state_dim = int(state_space.shape[0])
        self._image_extractor = FZeroXObservationCnnExtractor(
            image_space,
            features_dim=features_dim,
            conv_profile=conv_profile,
            layer_norm=False,
        )
        self._state_mlp = nn.Sequential(
            nn.Linear(self._state_dim, resolved_state_features_dim),
            nn.ReLU(),
        )
        if resolved_fusion_features_dim is None:
            self._fusion_mlp: nn.Module = nn.Identity()
        else:
            # The fusion layer lets image and scalar-state features interact
            # before the recurrent core, while keeping the legacy concat path.
            self._fusion_mlp = nn.Sequential(
                nn.Linear(combined_features_dim, resolved_fusion_features_dim),
                nn.ReLU(),
            )
        self._layer_norm: nn.Module = (
            nn.LayerNorm(output_features_dim) if layer_norm else nn.Identity()
        )

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        image = observations.get("image")
        state = observations.get("state")
        if image is None or state is None:
            raise ValueError(f"{type(self).__name__} expects observation keys 'image' and 'state'")

        state_flat = torch.flatten(state.float(), start_dim=1)
        if state_flat.shape[1] != self._state_dim:
            raise ValueError(
                f"Unexpected state vector width for {type(self).__name__}: "
                f"got {state_flat.shape[1]}, expected {self._state_dim}"
            )

        combined_features = torch.cat(
            [self._image_extractor(image), self._state_mlp(state_flat)],
            dim=1,
        )
        return self._layer_norm(self._fusion_mlp(combined_features))


class _ImageGeometry:
    def __init__(self, *, height: int, width: int, channels: int, conv_spec: ConvSpec) -> None:
        self.height = height
        self.width = width
        self.channels = channels
        self.conv_spec = conv_spec


def _resolve_supported_image_geometry(
    observation_space: spaces.Box,
    *,
    extractor_name: str,
    conv_profile: ConvProfile = "auto",
) -> _ImageGeometry:
    if len(observation_space.shape) != 3:
        raise ValueError(
            f"{extractor_name} expects a 3D image observation space, "
            f"got {observation_space.shape!r}"
        )

    if is_image_space_channels_first(observation_space):
        channels, height, width = observation_space.shape
    else:
        height, width, channels = observation_space.shape

    resolved_height = int(height)
    resolved_width = int(width)
    geometry = (resolved_height, resolved_width)
    try:
        conv_spec = _resolve_conv_spec(geometry, conv_profile=conv_profile)
    except KeyError as error:
        supported = ", ".join(f"{height}x{width}" for height, width in SUPPORTED_POLICY_GEOMETRIES)
        raise ValueError(
            f"Unsupported observation geometry {resolved_height}x{resolved_width} for "
            f"{extractor_name}; supported presets: {supported}"
        ) from error

    return _ImageGeometry(
        height=resolved_height,
        width=resolved_width,
        channels=int(channels),
        conv_spec=conv_spec,
    )


def _resolve_conv_spec(
    geometry: tuple[int, int],
    *,
    conv_profile: ConvProfile,
) -> ConvSpec:
    if conv_profile == "auto":
        return SUPPORTED_POLICY_GEOMETRIES[geometry]
    if conv_profile == "nature":
        return NATURE_CNN_CONV_SPEC
    if conv_profile == "compact_deep":
        return COMPACT_DEEP_CONV_SPEC
    if conv_profile == "compact_bottleneck":
        return COMPACT_BOTTLENECK_CONV_SPEC
    raise ValueError(f"Unsupported CNN conv profile: {conv_profile!r}")


def _image_flatten_dim(
    observation_space: spaces.Box,
    *,
    conv_profile: ConvProfile = "auto",
) -> int:
    geometry = _resolve_supported_image_geometry(
        observation_space,
        extractor_name=FZeroXImageStateExtractor.__name__,
        conv_profile=conv_profile,
    )
    height = geometry.height
    width = geometry.width
    output_channels = geometry.channels
    for out_channels, kernel_size, stride in geometry.conv_spec:
        output_channels = out_channels
        height = _conv_output_size(height, kernel_size[0], stride[0])
        width = _conv_output_size(width, kernel_size[1], stride[1])
    return output_channels * height * width


def _conv_output_size(input_size: int, kernel_size: int, stride: int) -> int:
    return ((input_size - kernel_size) // stride) + 1
