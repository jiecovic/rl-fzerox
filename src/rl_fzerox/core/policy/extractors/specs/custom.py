# src/rl_fzerox/core/policy/extractors/specs/custom.py
"""Custom CNN layer parsing for manager and runtime policy config.

Input arrives as small mapping objects from Pydantic models. The parser converts
them into ``ConvLayerSpec`` entries and validates layer geometry before network
construction.
"""

from __future__ import annotations

from rl_fzerox.core.domain.policy import (
    CnnActivationName,
    CnnLayerKind,
    is_activation_cnn_layer,
    is_pooling_cnn_layer,
    normalize_cnn_layer_kind,
    validate_cnn_layer_geometry,
)
from rl_fzerox.core.policy.extractors.specs.models import (
    ConvLayerSpec,
    ConvSpec,
    CustomConvLayerConfig,
)


def custom_conv_spec(
    custom_conv_layers: tuple[CustomConvLayerConfig, ...] | None,
) -> ConvSpec:
    if not custom_conv_layers:
        raise ValueError("custom CNN profile requires at least one convolution layer")

    return tuple(_custom_cnn_layer(layer) for layer in custom_conv_layers)


def _custom_cnn_layer(layer: CustomConvLayerConfig) -> ConvLayerSpec:
    kind = _custom_cnn_layer_kind(layer.get("kind"))
    if is_activation_cnn_layer(kind):
        return ConvLayerSpec(
            kind="activation",
            out_channels=_custom_cnn_layer_int(layer, "out_channels", default=1),
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            post_activation=True,
            activation=_custom_cnn_layer_activation(layer),
        )
    kernel_size = _custom_cnn_layer_int(layer, "kernel_size")
    stride = _custom_cnn_layer_int(layer, "stride")
    padding = _custom_cnn_layer_int(layer, "padding", default=0)
    validate_cnn_layer_geometry(
        kind=kind,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
    )
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


def _custom_cnn_layer_activation(layer: CustomConvLayerConfig) -> CnnActivationName:
    value = layer.get("activation")
    if value is None or value == "relu":
        return "relu"
    if value == "gelu":
        return "gelu"
    raise ValueError("custom CNN activation layers support activation='relu' or 'gelu'")


def _custom_cnn_layer_kind(value: object) -> CnnLayerKind:
    if value is None:
        return "conv"
    return normalize_cnn_layer_kind(value)
