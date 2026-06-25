# src/rl_fzerox/core/policy/extractors/__init__.py
"""Public facade for policy extractors, CNN blocks, and convolution specs.

Manager previews, runtime model builders, and tests import this package surface.
Implementation files stay split by network construction, blocks, and spec
resolution.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

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

if TYPE_CHECKING:
    from rl_fzerox.core.policy.extractors.blocks import (
        PostActivationResidualConvBlock,
        PreActivationResidualConvBlock,
    )
    from rl_fzerox.core.policy.extractors.networks import (
        FZeroXImageStateExtractor,
        FZeroXObservationCnnExtractor,
    )

_LAZY_EXPORT_MODULES = {
    "FZeroXImageStateExtractor": "rl_fzerox.core.policy.extractors.networks",
    "FZeroXObservationCnnExtractor": "rl_fzerox.core.policy.extractors.networks",
    "PostActivationResidualConvBlock": "rl_fzerox.core.policy.extractors.blocks",
    "PreActivationResidualConvBlock": "rl_fzerox.core.policy.extractors.blocks",
}

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


def __getattr__(name: str) -> object:
    module_name = _LAZY_EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(__all__)
