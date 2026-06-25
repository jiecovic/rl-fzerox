# src/rl_fzerox/core/policy/extractors/specs/geometry.py
"""CNN output-shape math and image-space validation for policy extractors.

Extractor builders and manager previews use this module to resolve image layout,
check that a conv stack fits, and compute flattened CNN dimensions.
"""

from __future__ import annotations

from gymnasium import spaces
from stable_baselines3.common.preprocessing import is_image_space_channels_first

from rl_fzerox.core.domain.policy import (
    CnnLayerKind,
    is_activation_cnn_layer,
    is_pooling_cnn_layer,
    is_residual_cnn_layer,
)
from rl_fzerox.core.policy.extractors.specs.models import (
    ConvLayerSpec,
    ConvProfile,
    ConvSpec,
    CustomConvLayerConfig,
    ImageGeometry,
)
from rl_fzerox.core.policy.extractors.specs.resolver import resolve_conv_spec


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
    conv_spec = resolve_conv_spec(
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


def cnn_layer_name(index: int, kind: CnnLayerKind) -> str:
    if kind == "activation":
        return f"act{index}"
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
    if is_pooling_cnn_layer(layer_spec.kind) or is_activation_cnn_layer(layer_spec.kind):
        return input_channels
    return layer_spec.out_channels


def cnn_layer_output_shape(
    *,
    height: int,
    width: int,
    layer_spec: ConvLayerSpec,
) -> tuple[int, int]:
    if is_activation_cnn_layer(layer_spec.kind):
        return height, width

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
