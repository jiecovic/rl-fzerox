# src/rl_fzerox/core/policy/extractors/specs/resolver.py
"""CNN profile resolver for built-in and custom convolution specs.

Runtime builders call this boundary with a profile name and optional custom
layer config. The returned ``ConvSpec`` is ready for geometry validation and
network construction.
"""

from __future__ import annotations

from rl_fzerox.core.policy.extractors.specs.custom import custom_conv_spec
from rl_fzerox.core.policy.extractors.specs.models import (
    ConvProfile,
    ConvSpec,
    CustomConvLayerConfig,
)
from rl_fzerox.core.policy.extractors.specs.profiles import (
    IMPALA_LARGE_CNN_CONV_SPEC,
    IMPALA_SMALL_CNN_CONV_SPEC,
    NATURE_CNN_CONV_SPEC,
)


def resolve_conv_spec(
    geometry: tuple[int, int],
    *,
    conv_profile: ConvProfile,
    custom_conv_layers: tuple[CustomConvLayerConfig, ...] | None = None,
) -> ConvSpec:
    """Return the convolution stack used by the policy extractor for one geometry."""

    if conv_profile == "nature":
        return NATURE_CNN_CONV_SPEC
    if conv_profile == "impala_small":
        return IMPALA_SMALL_CNN_CONV_SPEC
    if conv_profile == "impala_large":
        return IMPALA_LARGE_CNN_CONV_SPEC
    if conv_profile == "custom":
        return custom_conv_spec(custom_conv_layers)
    raise ValueError(f"Unsupported CNN conv profile: {conv_profile!r}")
