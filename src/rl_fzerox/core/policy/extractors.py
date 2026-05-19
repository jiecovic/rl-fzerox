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

from rl_fzerox.core.domain.cnn import (
    CnnLayerKind,
    is_pooling_cnn_layer,
    is_residual_cnn_layer,
    normalize_cnn_layer_kind,
    validate_cnn_layer_geometry,
)

ConvProfile = Literal[
    "nature",
    "impala_small",
    "impala_large",
    "custom",
]


@dataclass(frozen=True, slots=True)
class ConvLayerSpec:
    """One top-level CNN image layer in a supported policy image profile."""

    kind: CnnLayerKind
    out_channels: int
    kernel_size: tuple[int, int]
    stride: tuple[int, int]
    padding: tuple[int, int]
    post_activation: bool = True


ConvSpec = tuple[ConvLayerSpec, ...]
CustomConvLayerConfig = Mapping[str, object]


class PostActivationResidualConvBlock(nn.Module):
    """Residual block matching the custom-CNN block behavior used so far."""

    def __init__(
        self,
        in_channels: int,
        layer_spec: ConvLayerSpec,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            layer_spec.out_channels,
            kernel_size=layer_spec.kernel_size,
            stride=layer_spec.stride,
            padding=layer_spec.padding,
        )
        self.activation = nn.ReLU()
        self.conv2 = nn.Conv2d(
            layer_spec.out_channels,
            layer_spec.out_channels,
            kernel_size=layer_spec.kernel_size,
            stride=(1, 1),
            padding=layer_spec.padding,
        )
        self.projection = (
            nn.Conv2d(
                in_channels,
                layer_spec.out_channels,
                kernel_size=(1, 1),
                stride=layer_spec.stride,
                padding=(0, 0),
            )
            if in_channels != layer_spec.out_channels or layer_spec.stride != (1, 1)
            else nn.Identity()
        )
        self.output_activation = nn.ReLU()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = self.projection(inputs)
        output = self.activation(self.conv1(inputs))
        output = self.conv2(output)
        return self.output_activation(output + residual)


class PreActivationResidualConvBlock(nn.Module):
    """IMPALA-style residual block with an identity-preserving skip path."""

    def __init__(
        self,
        in_channels: int,
        layer_spec: ConvLayerSpec,
    ) -> None:
        super().__init__()
        self.activation1 = nn.ReLU()
        self.conv1 = nn.Conv2d(
            in_channels,
            layer_spec.out_channels,
            kernel_size=layer_spec.kernel_size,
            stride=layer_spec.stride,
            padding=layer_spec.padding,
        )
        self.activation2 = nn.ReLU()
        self.conv2 = nn.Conv2d(
            layer_spec.out_channels,
            layer_spec.out_channels,
            kernel_size=layer_spec.kernel_size,
            stride=(1, 1),
            padding=layer_spec.padding,
        )
        self.projection = (
            nn.Conv2d(
                in_channels,
                layer_spec.out_channels,
                kernel_size=(1, 1),
                stride=layer_spec.stride,
                padding=(0, 0),
            )
            if in_channels != layer_spec.out_channels or layer_spec.stride != (1, 1)
            else nn.Identity()
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = self.projection(inputs)
        output = self.conv1(self.activation1(inputs))
        output = self.conv2(self.activation2(output))
        return output + residual


def _conv_layer(
    out_channels: int,
    *,
    kernel_size: int,
    stride: int,
    padding: int = 0,
    post_activation: bool = True,
) -> ConvLayerSpec:
    return ConvLayerSpec(
        kind="conv",
        out_channels=out_channels,
        kernel_size=(kernel_size, kernel_size),
        stride=(stride, stride),
        padding=(padding, padding),
        post_activation=post_activation,
    )


def _pool_layer(
    *,
    kind: Literal["maxpool", "avgpool"],
    out_channels: int,
    kernel_size: int,
    stride: int,
    padding: int,
) -> ConvLayerSpec:
    return ConvLayerSpec(
        kind=kind,
        out_channels=out_channels,
        kernel_size=(kernel_size, kernel_size),
        stride=(stride, stride),
        padding=(padding, padding),
    )


def _residual_layer(
    out_channels: int,
    *,
    kind: CnnLayerKind = "residual_post",
    kernel_size: int = 3,
    stride: int = 1,
    padding: int = 1,
) -> ConvLayerSpec:
    return ConvLayerSpec(
        kind=kind,
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
IMPALA_SMALL_CNN_CONV_SPEC: ConvSpec = (
    _conv_layer(16, kernel_size=8, stride=4),
    _conv_layer(32, kernel_size=4, stride=2),
)


def _impala_conv_sequence(out_channels: int) -> ConvSpec:
    return (
        _conv_layer(out_channels, kernel_size=3, stride=1, padding=1, post_activation=False),
        _pool_layer(
            kind="maxpool",
            out_channels=out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
        ),
        _residual_layer(out_channels, kind="residual_pre"),
        _residual_layer(out_channels, kind="residual_pre"),
    )


IMPALA_LARGE_CNN_CONV_SPEC: ConvSpec = (
    _impala_conv_sequence(16) + _impala_conv_sequence(32) + _impala_conv_sequence(32)
)


class FZeroXObservationCnnExtractor(BaseFeaturesExtractor):
    """CNN extractor for supported F-Zero X observation image geometries."""

    def __init__(
        self,
        observation_space: spaces.Box,
        features_dim: int | Literal["auto"] = 512,
        conv_profile: ConvProfile = "nature",
        custom_conv_layers: tuple[CustomConvLayerConfig, ...] | None = None,
        custom_cnn_final_relu: bool = False,
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

        cnn_layers, capture_indices = self._build_conv_layers(
            input_channels=self._channels,
            conv_spec=self._conv_spec,
        )
        self._cnn_capture_indices = capture_indices
        if conv_profile == "impala_large" or (conv_profile == "custom" and custom_cnn_final_relu):
            cnn_layers.append(nn.ReLU())
        cnn = nn.Sequential(*cnn_layers, nn.Flatten())

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
    ) -> tuple[list[nn.Module], tuple[int, ...]]:
        layers: list[nn.Module] = []
        capture_indices: list[int] = []
        in_channels = input_channels
        for layer_spec in conv_spec:
            if layer_spec.kind == "conv":
                conv = nn.Conv2d(
                    in_channels,
                    layer_spec.out_channels,
                    kernel_size=layer_spec.kernel_size,
                    stride=layer_spec.stride,
                    padding=layer_spec.padding,
                )
                layers.append(conv)
                if layer_spec.post_activation:
                    layers.append(nn.ReLU())
            elif layer_spec.kind == "residual_pre":
                layers.append(PreActivationResidualConvBlock(in_channels, layer_spec))
            elif layer_spec.kind == "residual_post":
                layers.append(PostActivationResidualConvBlock(in_channels, layer_spec))
            elif layer_spec.kind == "maxpool":
                layers.append(_torch_pooling_layer(nn.MaxPool2d, layer_spec))
            elif layer_spec.kind == "avgpool":
                layers.append(_torch_pooling_layer(nn.AvgPool2d, layer_spec))
            else:
                raise ValueError(f"Unsupported CNN layer kind: {layer_spec.kind!r}")
            capture_indices.append(len(layers) - 1)
            in_channels = cnn_layer_output_channels(in_channels, layer_spec)
        return layers, tuple(capture_indices)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Convert either NHWC or NCHW observations into shared PPO features."""

        channels_first = self._channels_first_observations(observations)
        return self._layer_norm(self._linear(self._cnn(channels_first)))

    def convolution_activations(
        self,
        observations: torch.Tensor,
    ) -> tuple[tuple[str, torch.Tensor], ...]:
        """Return top-level image-layer outputs for CNN visualization."""

        activations: list[tuple[str, torch.Tensor]] = []
        output = self._channels_first_observations(observations)
        capture_positions = {
            module_index: layer_index
            for layer_index, module_index in enumerate(self._cnn_capture_indices)
        }
        for module_index, layer in enumerate(self._cnn):
            if isinstance(layer, nn.Flatten):
                break
            output = layer(output)
            layer_index = capture_positions.get(module_index)
            if layer_index is None:
                continue
            layer_spec = self._conv_spec[layer_index]
            activations.append((cnn_layer_name(layer_index + 1, layer_spec.kind), output.detach()))
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
        conv_profile: ConvProfile = "nature",
        custom_conv_layers: tuple[CustomConvLayerConfig, ...] | None = None,
        custom_cnn_final_relu: bool = False,
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
            custom_cnn_final_relu=custom_cnn_final_relu,
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


def _torch_pooling_layer(
    pool_type: type[nn.MaxPool2d] | type[nn.AvgPool2d],
    layer_spec: ConvLayerSpec,
) -> nn.Module:
    return pool_type(
        kernel_size=layer_spec.kernel_size,
        stride=layer_spec.stride,
        padding=layer_spec.padding,
    )


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
    conv_profile: ConvProfile = "nature",
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
    conv_spec = _resolve_conv_spec(
        (resolved_height, resolved_width),
        conv_profile=conv_profile,
        custom_conv_layers=custom_conv_layers,
    )

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
    if conv_profile == "nature":
        return NATURE_CNN_CONV_SPEC
    if conv_profile == "impala_small":
        return IMPALA_SMALL_CNN_CONV_SPEC
    if conv_profile == "impala_large":
        return IMPALA_LARGE_CNN_CONV_SPEC
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
        next_height, next_width = cnn_layer_output_shape(
            height=current_height,
            width=current_width,
            layer_spec=layer,
        )
        if next_height < 1 or next_width < 1:
            raise ValueError(
                f"CNN profile {profile_name!r} collapses {height}x{width} at "
                f"{cnn_layer_name(index, layer.kind)}: "
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

    return tuple(_custom_cnn_layer(layer) for layer in custom_conv_layers)


def _image_flatten_dim(
    observation_space: spaces.Box,
    *,
    conv_profile: ConvProfile = "nature",
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
        output_channels = cnn_layer_output_channels(output_channels, layer_spec)
        height, width = cnn_layer_output_shape(
            height=height,
            width=width,
            layer_spec=layer_spec,
        )
    return output_channels * height * width


def _custom_cnn_layer(layer: CustomConvLayerConfig) -> ConvLayerSpec:
    kind = _custom_cnn_layer_kind(layer.get("kind"))
    kernel_size = _custom_cnn_layer_int(layer, "kernel_size")
    stride = _custom_cnn_layer_int(layer, "stride")
    padding = _custom_cnn_layer_int(layer, "padding", default=0)
    validate_cnn_layer_geometry(kind=kind, kernel_size=kernel_size, padding=padding)
    return ConvLayerSpec(
        kind=kind,
        out_channels=_custom_cnn_layer_int(
            layer,
            "out_channels",
            default=1 if is_pooling_cnn_layer(kind) else None,
        ),
        kernel_size=(kernel_size, kernel_size),
        stride=(stride, stride),
        padding=(padding, padding),
        post_activation=_custom_cnn_layer_bool(layer, "post_activation", default=True),
    )


def _custom_cnn_layer_int(
    layer: CustomConvLayerConfig,
    key: str,
    *,
    default: int | None = None,
) -> int:
    if key in layer:
        value = layer[key]
    elif default is not None:
        value = default
    else:
        raise KeyError(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"custom CNN layer field {key!r} must be an integer")
    return value


def _custom_cnn_layer_bool(
    layer: CustomConvLayerConfig,
    key: str,
    *,
    default: bool,
) -> bool:
    if key not in layer:
        return default
    value = layer[key]
    if not isinstance(value, bool):
        raise TypeError(f"custom CNN layer field {key!r} must be a boolean")
    return value


def _custom_cnn_layer_kind(value: object) -> CnnLayerKind:
    if value is None:
        return "conv"
    return normalize_cnn_layer_kind(value)


def cnn_layer_name(index: int, kind: CnnLayerKind) -> str:
    if kind == "residual_pre":
        return f"res-pre{index}"
    if kind == "residual_post":
        return f"res{index}"
    if kind == "maxpool":
        return f"pool{index}"
    if kind == "avgpool":
        return f"avgpool{index}"
    return f"conv{index}"


def cnn_layer_output_channels(input_channels: int, layer_spec: ConvLayerSpec) -> int:
    return input_channels if is_pooling_cnn_layer(layer_spec.kind) else layer_spec.out_channels


def cnn_layer_output_shape(
    *,
    height: int,
    width: int,
    layer_spec: ConvLayerSpec,
) -> tuple[int, int]:
    first_height = _conv_output_size(
        height,
        layer_spec.kernel_size[0],
        layer_spec.stride[0],
        padding=layer_spec.padding[0],
    )
    first_width = _conv_output_size(
        width,
        layer_spec.kernel_size[1],
        layer_spec.stride[1],
        padding=layer_spec.padding[1],
    )
    if not is_residual_cnn_layer(layer_spec.kind) or first_height < 1 or first_width < 1:
        return first_height, first_width

    second_height = _conv_output_size(
        first_height,
        layer_spec.kernel_size[0],
        1,
        padding=layer_spec.padding[0],
    )
    second_width = _conv_output_size(
        first_width,
        layer_spec.kernel_size[1],
        1,
        padding=layer_spec.padding[1],
    )
    if (second_height, second_width) != (first_height, first_width):
        raise ValueError(
            "residual CNN blocks require the second convolution to preserve spatial size; "
            "use an odd kernel_size with padding=kernel_size//2"
        )
    return second_height, second_width


def _conv_output_size(input_size: int, kernel_size: int, stride: int, padding: int = 0) -> int:
    return ((input_size + (2 * padding) - kernel_size) // stride) + 1


def conv_output_size(input_size: int, kernel_size: int, stride: int, padding: int = 0) -> int:
    """Return the spatial output size for one valid convolution dimension."""

    return _conv_output_size(input_size, kernel_size, stride, padding)
