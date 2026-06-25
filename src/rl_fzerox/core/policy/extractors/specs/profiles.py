# src/rl_fzerox/core/policy/extractors/specs/profiles.py
"""Built-in CNN layer profiles used by policy image extractors.

The profiles define Nature-style, small IMPALA-style, and larger IMPALA-style
stacks as ``ConvLayerSpec`` tuples. Network construction reads these specs
without depending on how a profile was selected.
"""

from __future__ import annotations

from typing import Literal

from rl_fzerox.core.domain.policy import CnnActivationName, CnnLayerKind
from rl_fzerox.core.policy.extractors.specs.models import ConvLayerSpec, ConvSpec


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


def _activation_layer(
    *,
    out_channels: int,
    activation: CnnActivationName,
) -> ConvLayerSpec:
    return ConvLayerSpec(
        kind="activation",
        out_channels=out_channels,
        kernel_size=(1, 1),
        stride=(1, 1),
        padding=(0, 0),
        post_activation=True,
        activation=activation,
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
    _impala_conv_sequence(16)
    + _impala_conv_sequence(32)
    + _impala_conv_sequence(32)
    + (_activation_layer(out_channels=32, activation="relu"),)
)
