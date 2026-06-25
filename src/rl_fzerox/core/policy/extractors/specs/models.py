# src/rl_fzerox/core/policy/extractors/specs/models.py
"""Shared data models for policy CNN convolution specs.

The dataclasses describe the compact internal representation used by built-in
profiles, custom profile parsing, extractor construction, and architecture
previews.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from rl_fzerox.core.domain.policy import CnnActivationName, CnnLayerKind

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
    activation: CnnActivationName | None = None


ConvSpec = tuple[ConvLayerSpec, ...]
CustomConvLayerConfig = Mapping[str, object]


@dataclass(frozen=True, slots=True)
class ImageGeometry:
    height: int
    width: int
    channels: int
    conv_spec: ConvSpec
