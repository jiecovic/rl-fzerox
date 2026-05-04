# src/rl_fzerox/core/policy/extractors.py
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

import torch
from gymnasium import spaces
from stable_baselines3.common.preprocessing import is_image_space_channels_first
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn

ConvProfile = Literal[
    "auto",
    "nature",
    "nature_32_64_128",
    "nature_wide",
    "nature_extra_k3",
    "compact_deep",
    "compact_bottleneck",
    "tiny_256",
    "custom",
]


@dataclass(frozen=True, slots=True)
class ConvLayerSpec:
    """One CNN convolution layer in a supported policy image profile."""

    out_channels: int
    kernel_size: tuple[int, int]
    stride: tuple[int, int]
    padding: tuple[int, int]


ConvSpec = tuple[ConvLayerSpec, ...]
CustomConvLayerConfig = Mapping[str, int]


def _conv_layer(
    out_channels: int,
    *,
    kernel_size: int,
    stride: int,
    padding: int = 0,
) -> ConvLayerSpec:
    return ConvLayerSpec(
        out_channels=out_channels,
        kernel_size=(kernel_size, kernel_size),
        stride=(stride, stride),
        padding=(padding, padding),
    )


NATURE_CNN_CONV_SPEC: ConvSpec = (
    _conv_layer(32, kernel_size=8, stride=4),
    _conv_layer(64, kernel_size=4, stride=2),
    _conv_layer(64, kernel_size=3, stride=1),
)
NATURE_32_64_128_CONV_SPEC: ConvSpec = (
    _conv_layer(32, kernel_size=8, stride=4),
    _conv_layer(64, kernel_size=4, stride=2),
    _conv_layer(128, kernel_size=3, stride=1),
)
NATURE_WIDE_CONV_SPEC: ConvSpec = (
    _conv_layer(64, kernel_size=8, stride=4),
    _conv_layer(128, kernel_size=4, stride=2),
    _conv_layer(128, kernel_size=3, stride=1),
)
NATURE_EXTRA_K3_CONV_SPEC: ConvSpec = (
    *NATURE_CNN_CONV_SPEC,
    _conv_layer(64, kernel_size=3, stride=1),
)
LEGACY_DEEP_CONV_SPEC: ConvSpec = (
    _conv_layer(64, kernel_size=8, stride=4),
    _conv_layer(64, kernel_size=4, stride=2),
    _conv_layer(128, kernel_size=3, stride=2),
    _conv_layer(128, kernel_size=3, stride=1),
)
COMPACT_DEEP_CONV_SPEC: ConvSpec = (
    _conv_layer(64, kernel_size=8, stride=2),
    _conv_layer(64, kernel_size=4, stride=2),
    _conv_layer(128, kernel_size=4, stride=2),
    _conv_layer(128, kernel_size=4, stride=2),
)
COMPACT_BOTTLENECK_CONV_SPEC: ConvSpec = (
    _conv_layer(64, kernel_size=8, stride=2),
    _conv_layer(64, kernel_size=8, stride=2),
    _conv_layer(64, kernel_size=4, stride=2),
    _conv_layer(128, kernel_size=3, stride=2),
    _conv_layer(256, kernel_size=4, stride=1),
)
TINY_256_CONV_SPEC: ConvSpec = (
    _conv_layer(64, kernel_size=8, stride=2),
    _conv_layer(64, kernel_size=4, stride=2),
    _conv_layer(128, kernel_size=4, stride=2),
    _conv_layer(256, kernel_size=4, stride=2),
)
SUPPORTED_POLICY_GEOMETRIES: dict[tuple[int, int], ConvSpec] = {
    (84, 116): NATURE_CNN_CONV_SPEC,
    (92, 124): NATURE_CNN_CONV_SPEC,
    (116, 164): LEGACY_DEEP_CONV_SPEC,
    (98, 130): COMPACT_DEEP_CONV_SPEC,
    (66, 82): COMPACT_DEEP_CONV_SPEC,
    (60, 76): NATURE_CNN_CONV_SPEC,
    (68, 68): NATURE_CNN_CONV_SPEC,
    (84, 84): NATURE_CNN_CONV_SPEC,
    (76, 100): NATURE_CNN_CONV_SPEC,
    (64, 64): TINY_256_CONV_SPEC,
}


class FZeroXObservationCnnExtractor(BaseFeaturesExtractor):
    """CNN extractor for the supported aspect-corrected F-Zero X observation presets."""

    def __init__(
        self,
        observation_space: spaces.Box,
        features_dim: int | Literal["auto"] = 512,
        conv_profile: ConvProfile = "auto",
        custom_conv_layers: tuple[CustomConvLayerConfig, ...] | None = None,
        layer_norm: bool = False,
    ) -> None:
        image_geometry = _resolve_supported_image_geometry(
            observation_space,
            extractor_name=type(self).__name__,
            conv_profile=conv_profile,
            custom_conv_layers=custom_conv_layers,
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
        for layer_spec in conv_spec:
            layers.append(
                nn.Conv2d(
                    in_channels,
                    layer_spec.out_channels,
                    kernel_size=layer_spec.kernel_size,
                    stride=layer_spec.stride,
                    padding=layer_spec.padding,
                )
            )
            layers.append(nn.ReLU())
            in_channels = layer_spec.out_channels
        return layers

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Convert either NHWC or NCHW observations into shared PPO features."""

        channels_first = self._channels_first_observations(observations)
        return self._layer_norm(self._linear(self._cnn(channels_first)))

    def convolution_activations(
        self,
        observations: torch.Tensor,
    ) -> tuple[tuple[str, torch.Tensor], ...]:
        """Return post-ReLU activation maps for each convolutional layer."""

        activations: list[tuple[str, torch.Tensor]] = []
        output = self._channels_first_observations(observations)
        conv_index = 0
        for layer in self._cnn:
            output = layer(output)
            if isinstance(layer, nn.ReLU):
                conv_index += 1
                activations.append((f"conv{conv_index}", output.detach()))
        return tuple(activations)

    def _channels_first_observations(self, observations: torch.Tensor) -> torch.Tensor:
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

        return channels_first.float()


class FZeroXImageStateExtractor(BaseFeaturesExtractor):
    """CNN image features plus a scalar-state branch for Dict observations."""

    def __init__(
        self,
        observation_space: spaces.Dict,
        features_dim: int | Literal["auto"] = 512,
        state_features_dim: int = 64,
        state_net_arch: tuple[int, ...] | None = None,
        fusion_features_dim: int | None = None,
        conv_profile: ConvProfile = "auto",
        custom_conv_layers: tuple[CustomConvLayerConfig, ...] | None = None,
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

        state_dim = int(state_space.shape[0])
        image_features_dim = (
            _image_flatten_dim(
                image_space,
                conv_profile=conv_profile,
                custom_conv_layers=custom_conv_layers,
            )
            if features_dim == "auto"
            else int(features_dim)
        )
        resolved_state_net_arch = (
            (int(state_features_dim),)
            if state_net_arch is None
            else tuple(int(width) for width in state_net_arch)
        )
        state_branch_output_dim = (
            state_dim if not resolved_state_net_arch else resolved_state_net_arch[-1]
        )
        combined_features_dim = image_features_dim + state_branch_output_dim
        resolved_fusion_features_dim = (
            None if fusion_features_dim is None else int(fusion_features_dim)
        )
        output_features_dim = (
            combined_features_dim
            if resolved_fusion_features_dim is None
            else resolved_fusion_features_dim
        )
        super().__init__(observation_space, output_features_dim)

        self._state_dim = state_dim
        self._image_extractor = FZeroXObservationCnnExtractor(
            image_space,
            features_dim=features_dim,
            conv_profile=conv_profile,
            custom_conv_layers=custom_conv_layers,
            layer_norm=False,
        )
        self._state_mlp = _state_branch_mlp(
            input_dim=self._state_dim,
            widths=resolved_state_net_arch,
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


def _state_branch_mlp(*, input_dim: int, widths: tuple[int, ...]) -> nn.Module:
    if not widths:
        return nn.Identity()

    layers: list[nn.Module] = []
    previous_dim = input_dim
    for width in widths:
        layers.append(nn.Linear(previous_dim, int(width)))
        layers.append(nn.ReLU())
        previous_dim = int(width)
    return nn.Sequential(*layers)


@dataclass(frozen=True, slots=True)
class _ImageGeometry:
    height: int
    width: int
    channels: int
    conv_spec: ConvSpec


def _resolve_supported_image_geometry(
    observation_space: spaces.Box,
    *,
    extractor_name: str,
    conv_profile: ConvProfile = "auto",
    custom_conv_layers: tuple[CustomConvLayerConfig, ...] | None = None,
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
        conv_spec = _resolve_conv_spec(
            geometry,
            conv_profile=conv_profile,
            custom_conv_layers=custom_conv_layers,
        )
    except KeyError as error:
        supported = ", ".join(f"{height}x{width}" for height, width in SUPPORTED_POLICY_GEOMETRIES)
        raise ValueError(
            f"Unsupported observation geometry {resolved_height}x{resolved_width} for "
            f"{extractor_name}; supported presets: {supported}"
        ) from error

    ensure_conv_spec_fits_geometry(
        height=resolved_height,
        width=resolved_width,
        conv_spec=conv_spec,
        profile_name=conv_profile,
    )

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
    custom_conv_layers: tuple[CustomConvLayerConfig, ...] | None = None,
) -> ConvSpec:
    if conv_profile == "auto":
        return SUPPORTED_POLICY_GEOMETRIES[geometry]
    if conv_profile == "nature":
        return NATURE_CNN_CONV_SPEC
    if conv_profile == "nature_32_64_128":
        return NATURE_32_64_128_CONV_SPEC
    if conv_profile == "nature_wide":
        return NATURE_WIDE_CONV_SPEC
    if conv_profile == "nature_extra_k3":
        return NATURE_EXTRA_K3_CONV_SPEC
    if conv_profile == "compact_deep":
        return COMPACT_DEEP_CONV_SPEC
    if conv_profile == "compact_bottleneck":
        return COMPACT_BOTTLENECK_CONV_SPEC
    if conv_profile == "tiny_256":
        return TINY_256_CONV_SPEC
    if conv_profile == "custom":
        return _custom_conv_spec(custom_conv_layers)
    raise ValueError(f"Unsupported CNN conv profile: {conv_profile!r}")


def resolve_conv_spec(
    geometry: tuple[int, int],
    *,
    conv_profile: ConvProfile,
    custom_conv_layers: tuple[CustomConvLayerConfig, ...] | None = None,
) -> ConvSpec:
    """Return the convolution stack used by the policy extractor for one geometry."""

    return _resolve_conv_spec(
        geometry,
        conv_profile=conv_profile,
        custom_conv_layers=custom_conv_layers,
    )


def ensure_conv_spec_fits_geometry(
    *,
    height: int,
    width: int,
    conv_spec: ConvSpec,
    profile_name: str,
) -> None:
    current_height = height
    current_width = width
    for index, layer in enumerate(conv_spec, start=1):
        next_height = _conv_output_size(
            current_height,
            layer.kernel_size[0],
            layer.stride[0],
            padding=layer.padding[0],
        )
        next_width = _conv_output_size(
            current_width,
            layer.kernel_size[1],
            layer.stride[1],
            padding=layer.padding[1],
        )
        if next_height < 1 or next_width < 1:
            raise ValueError(
                f"CNN profile {profile_name!r} collapses {height}x{width} at conv{index}: "
                f"{current_height}x{current_width} with k={layer.kernel_size[0]} "
                f"s={layer.stride[0]} p={layer.padding[0]}"
            )
        current_height = next_height
        current_width = next_width


def _custom_conv_spec(
    custom_conv_layers: tuple[CustomConvLayerConfig, ...] | None,
) -> ConvSpec:
    if not custom_conv_layers:
        raise ValueError("custom CNN profile requires at least one convolution layer")

    return tuple(
        _conv_layer(
            int(layer["out_channels"]),
            kernel_size=int(layer["kernel_size"]),
            stride=int(layer["stride"]),
            padding=int(layer.get("padding", 0)),
        )
        for layer in custom_conv_layers
    )


def _image_flatten_dim(
    observation_space: spaces.Box,
    *,
    conv_profile: ConvProfile = "auto",
    custom_conv_layers: tuple[CustomConvLayerConfig, ...] | None = None,
) -> int:
    geometry = _resolve_supported_image_geometry(
        observation_space,
        extractor_name=FZeroXImageStateExtractor.__name__,
        conv_profile=conv_profile,
        custom_conv_layers=custom_conv_layers,
    )
    height = geometry.height
    width = geometry.width
    output_channels = geometry.channels
    for layer_spec in geometry.conv_spec:
        output_channels = layer_spec.out_channels
        height = _conv_output_size(
            height,
            layer_spec.kernel_size[0],
            layer_spec.stride[0],
            padding=layer_spec.padding[0],
        )
        width = _conv_output_size(
            width,
            layer_spec.kernel_size[1],
            layer_spec.stride[1],
            padding=layer_spec.padding[1],
        )
    return output_channels * height * width


def _conv_output_size(input_size: int, kernel_size: int, stride: int, padding: int = 0) -> int:
    return ((input_size + (2 * padding) - kernel_size) // stride) + 1


def conv_output_size(input_size: int, kernel_size: int, stride: int, padding: int = 0) -> int:
    """Return the spatial output size for one valid convolution dimension."""

    return _conv_output_size(input_size, kernel_size, stride, padding)
