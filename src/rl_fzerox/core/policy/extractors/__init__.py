# src/rl_fzerox/core/policy/extractors/__init__.py
from __future__ import annotations

from rl_fzerox.core.policy.extractors.blocks import (
    PostActivationResidualConvBlock,
    PreActivationResidualConvBlock,
)
from rl_fzerox.core.policy.extractors.networks import (
    FZeroXImageStateExtractor,
    FZeroXObservationCnnExtractor,
)
from rl_fzerox.core.policy.extractors.specs import (
    IMPALA_LARGE_CNN_CONV_SPEC,
    IMPALA_SMALL_CNN_CONV_SPEC,
    NATURE_CNN_CONV_SPEC,
    ConvLayerSpec,
    ConvProfile,
    ConvSpec,
    CustomConvLayerConfig,
    cnn_layer_name,
    cnn_layer_output_channels,
    cnn_layer_output_shape,
    conv_output_size,
    ensure_conv_spec_fits_geometry,
    resolve_conv_spec,
)

__all__ = [
    "IMPALA_LARGE_CNN_CONV_SPEC",
    "IMPALA_SMALL_CNN_CONV_SPEC",
    "NATURE_CNN_CONV_SPEC",
    "ConvLayerSpec",
    "ConvProfile",
    "ConvSpec",
    "CustomConvLayerConfig",
    "FZeroXImageStateExtractor",
    "FZeroXObservationCnnExtractor",
    "PostActivationResidualConvBlock",
    "PreActivationResidualConvBlock",
    "cnn_layer_name",
    "cnn_layer_output_channels",
    "cnn_layer_output_shape",
    "conv_output_size",
    "ensure_conv_spec_fits_geometry",
    "resolve_conv_spec",
]
