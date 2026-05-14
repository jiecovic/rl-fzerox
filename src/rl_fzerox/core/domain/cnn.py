# src/rl_fzerox/core/domain/cnn.py
from __future__ import annotations

from typing import Literal, TypeAlias

CnnLayerKind: TypeAlias = Literal["conv", "residual"]


def residual_padding_for_kernel(kernel_size: int) -> int:
    """Return symmetric same-padding for a residual block's internal convolutions."""

    if kernel_size <= 0:
        raise ValueError("CNN kernel_size must be positive")
    if kernel_size % 2 == 0:
        raise ValueError("residual CNN blocks require an odd kernel_size")
    return kernel_size // 2


def validate_residual_cnn_padding(*, kind: CnnLayerKind, kernel_size: int, padding: int) -> None:
    if kind != "residual":
        return
    expected_padding = residual_padding_for_kernel(kernel_size)
    if padding != expected_padding:
        raise ValueError(
            "residual CNN blocks require padding=kernel_size//2 so the skip path matches"
        )
