# src/rl_fzerox/core/policy/extractors/specs.py
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from gymnasium import spaces
from stable_baselines3.common.preprocessing import is_image_space_channels_first

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


@dataclass(frozen=True, slots=True)
class ImageGeometry:
    height: int
    width: int
    channels: int
    conv_spec: ConvSpec


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


def resolve_supported_image_geometry(
    observation_space: spaces.Box,
    *,
    extractor_name: str,
    conv_profile: ConvProfile = "nature",
    custom_conv_layers: tuple[CustomConvLayerConfig, ...] | None = None,
) -> ImageGeometry:
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

    return ImageGeometry(
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


def image_flatten_dim(
    observation_space: spaces.Box,
    *,
    extractor_name: str,
    conv_profile: ConvProfile = "nature",
    custom_conv_layers: tuple[CustomConvLayerConfig, ...] | None = None,
) -> int:
    geometry = resolve_supported_image_geometry(
        observation_space,
        extractor_name=extractor_name,
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


def _custom_conv_spec(
    custom_conv_layers: tuple[CustomConvLayerConfig, ...] | None,
) -> ConvSpec:
    if not custom_conv_layers:
        raise ValueError("custom CNN profile requires at least one convolution layer")

    return tuple(_custom_cnn_layer(layer) for layer in custom_conv_layers)


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
