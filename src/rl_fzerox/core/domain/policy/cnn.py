# src/rl_fzerox/core/domain/policy/cnn.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

type CnnActivationName = Literal["relu", "gelu"]

type CnnLayerKind = Literal[
    "conv",
    "residual_pre",
    "residual_post",
    "maxpool",
    "avgpool",
    "activation",
]


@dataclass(frozen=True, slots=True)
class _CnnLayerKindSpec:
    value: CnnLayerKind
    aliases: tuple[str, ...] = ()
    residual: bool = False
    pooling: bool = False
    activation: bool = False


_cnn_layer_kind_specs = (
    _CnnLayerKindSpec("conv"),
    _CnnLayerKindSpec("residual_pre", residual=True),
    _CnnLayerKindSpec("residual_post", aliases=("residual",), residual=True),
    _CnnLayerKindSpec("maxpool", pooling=True),
    _CnnLayerKindSpec("avgpool", pooling=True),
    _CnnLayerKindSpec("activation", activation=True),
)


def normalize_cnn_layer_kind(value: object) -> CnnLayerKind:
    """Return the canonical layer kind stored by new configs."""

    return _cnn_layer_kind_spec(value).value


def is_residual_cnn_layer(kind: CnnLayerKind) -> bool:
    return _cnn_layer_kind_spec(kind).residual


def is_pooling_cnn_layer(kind: CnnLayerKind) -> bool:
    return _cnn_layer_kind_spec(kind).pooling


def is_activation_cnn_layer(kind: CnnLayerKind) -> bool:
    return _cnn_layer_kind_spec(kind).activation


def _cnn_layer_kind_spec(value: object) -> _CnnLayerKindSpec:
    for spec in _cnn_layer_kind_specs:
        if value == spec.value or value in spec.aliases:
            return spec
    raise ValueError(f"Unsupported CNN layer kind: {value!r}")


def residual_padding_for_kernel(kernel_size: int) -> int:
    """Return symmetric same-padding for a residual block's internal convolutions."""

    if kernel_size <= 0:
        raise ValueError("CNN kernel_size must be positive")
    if kernel_size % 2 == 0:
        raise ValueError("residual CNN blocks require an odd kernel_size")
    return kernel_size // 2


def validate_cnn_layer_geometry(
    *,
    kind: CnnLayerKind,
    kernel_size: int,
    stride: int,
    padding: int,
) -> None:
    """Validate layer-kind-specific shape constraints before PyTorch builds modules."""

    if is_activation_cnn_layer(kind):
        if kernel_size != 1 or stride != 1 or padding != 0:
            raise ValueError("activation CNN layers require kernel_size=1, stride=1, padding=0")
        return
    validate_residual_cnn_padding(kind=kind, kernel_size=kernel_size, padding=padding)
    if is_pooling_cnn_layer(kind) and padding > kernel_size // 2:
        raise ValueError("pooling CNN layers require padding<=kernel_size//2")


def validate_residual_cnn_padding(*, kind: CnnLayerKind, kernel_size: int, padding: int) -> None:
    if not is_residual_cnn_layer(kind):
        return
    expected_padding = residual_padding_for_kernel(kernel_size)
    if padding != expected_padding:
        raise ValueError(
            "residual CNN blocks require padding=kernel_size//2 so the skip path matches"
        )
