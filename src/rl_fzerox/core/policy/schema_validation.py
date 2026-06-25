# src/rl_fzerox/core/policy/schema_validation.py
"""Shared validation rules for manager and runtime policy schemas.

Manager run specs and runtime configs keep separate Pydantic models. This module
holds the policy-specific CNN and auxiliary-state validation rules they share.
"""

from __future__ import annotations

from collections.abc import Sequence

from rl_fzerox.core.domain.policy import (
    CnnActivationName,
    CnnLayerKind,
    is_activation_cnn_layer,
    normalize_cnn_layer_kind,
    validate_cnn_layer_geometry,
)
from rl_fzerox.core.policy.auxiliary_state.names import AuxiliaryStateTargetName
from rl_fzerox.core.policy.auxiliary_state.targets import (
    auxiliary_state_target_supports_grounded_only,
)


def normalize_policy_cnn_layer_kind(value: object) -> CnnLayerKind:
    return normalize_cnn_layer_kind(value)


def validate_policy_cnn_layer_geometry(
    *,
    kind: CnnLayerKind,
    kernel_size: int,
    stride: int,
    padding: int,
) -> None:
    validate_cnn_layer_geometry(
        kind=kind,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
    )


def default_policy_cnn_layer_activation(
    *,
    kind: CnnLayerKind,
    activation: CnnActivationName | None,
) -> CnnActivationName | None:
    if is_activation_cnn_layer(kind) and activation is None:
        return "relu"
    return activation


def validate_policy_custom_conv_layers_present(
    *,
    conv_profile: str,
    has_custom_layers: bool,
    message: str,
) -> None:
    if conv_profile == "custom" and not has_custom_layers:
        raise ValueError(message)


def validate_policy_auxiliary_grounded_only(
    *,
    name: AuxiliaryStateTargetName,
    grounded_only: bool,
) -> None:
    if grounded_only and not auxiliary_state_target_supports_grounded_only(name):
        raise ValueError("grounded_only is not supported for this auxiliary-state target")


def validate_policy_auxiliary_losses(
    *,
    loss_names: Sequence[AuxiliaryStateTargetName],
    enabled: bool,
    duplicate_message: str,
    disabled_message: str,
) -> None:
    if len(set(loss_names)) != len(loss_names):
        raise ValueError(duplicate_message)
    if loss_names and not enabled:
        raise ValueError(disabled_message)
