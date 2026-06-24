# src/rl_fzerox/core/policy/activations.py
from __future__ import annotations

from typing import Literal

type ActivationName = Literal["relu", "tanh", "gelu"]


def resolve_policy_activation_fn(name: str):
    """Map a configured policy activation name to a torch module class."""

    from torch import nn

    if name == "relu":
        return nn.ReLU
    if name == "tanh":
        return nn.Tanh
    if name == "gelu":
        return nn.GELU
    raise ValueError(f"Unsupported policy activation: {name!r}")
