# src/rl_fzerox/core/policy/activations.py
"""Activation-name vocabulary for policy network configuration.

Runtime and manager schemas store compact string names. Extractor and policy
builders call this module when they need the matching torch activation class.
"""

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
