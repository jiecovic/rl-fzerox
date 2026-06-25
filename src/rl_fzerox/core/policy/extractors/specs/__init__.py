# src/rl_fzerox/core/policy/extractors/specs/__init__.py
"""Facade for CNN profile specs, custom parsing, and geometry validation.

Callers keep importing ``policy.extractors.specs`` while the implementation is
split into profiles, custom layer parsing, shape math, and shared data models.
"""

from __future__ import annotations

from rl_fzerox.core.policy.extractors.specs.geometry import (
    cnn_layer_name,
    cnn_layer_output_channels,
    cnn_layer_output_shape,
    conv_output_size,
    ensure_conv_spec_fits_geometry,
    image_flatten_dim,
    resolve_supported_image_geometry,
)
from rl_fzerox.core.policy.extractors.specs.models import (
    ConvLayerSpec,
    ConvProfile,
    ConvSpec,
    CustomConvLayerConfig,
    ImageGeometry,
)
from rl_fzerox.core.policy.extractors.specs.profiles import (
    IMPALA_LARGE_CNN_CONV_SPEC,
    IMPALA_SMALL_CNN_CONV_SPEC,
    NATURE_CNN_CONV_SPEC,
)
from rl_fzerox.core.policy.extractors.specs.resolver import resolve_conv_spec

__all__ = [
    "IMPALA_LARGE_CNN_CONV_SPEC",
    "IMPALA_SMALL_CNN_CONV_SPEC",
    "NATURE_CNN_CONV_SPEC",
    "ConvLayerSpec",
    "ConvProfile",
    "ConvSpec",
    "CustomConvLayerConfig",
    "ImageGeometry",
    "cnn_layer_name",
    "cnn_layer_output_channels",
    "cnn_layer_output_shape",
    "conv_output_size",
    "ensure_conv_spec_fits_geometry",
    "image_flatten_dim",
    "resolve_conv_spec",
    "resolve_supported_image_geometry",
]
