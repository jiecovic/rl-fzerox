# src/rl_fzerox/core/policy/auxiliary_state/actor_regularization/losses.py
"""Pure tensor losses shared by actor regularization paths.

The functions operate on already-resolved tensors and sample masks. They stay
free of policy and distribution state so the mixin and metric code can reuse
them directly.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class _SignedBalanceLoss:
    bias: torch.Tensor
    loss_value: torch.Tensor
    total_loss: torch.Tensor


def _signed_balance_loss(
    values: torch.Tensor,
    *,
    deadzone: float,
    loss_weight: float,
    sample_mask: torch.Tensor | None,
) -> _SignedBalanceLoss | None:
    if loss_weight <= 0.0:
        return None
    bias, has_active_samples = _masked_mean(values, sample_mask)
    if not has_active_samples:
        return None
    loss_value = (bias.abs() - values.new_tensor(deadzone)).relu().square()
    return _SignedBalanceLoss(
        bias=bias,
        loss_value=loss_value,
        total_loss=loss_weight * loss_value,
    )


def _std_cap_loss(
    values: torch.Tensor,
    *,
    cap: float,
    sample_mask: torch.Tensor | None,
) -> torch.Tensor | None:
    per_sample = (values - values.new_tensor(cap)).relu().square()
    loss_value, has_active_samples = _masked_mean(per_sample, sample_mask)
    if not has_active_samples:
        return None
    return loss_value


def _combined_mask(
    scope_mask: torch.Tensor,
    sample_mask: torch.Tensor | None,
) -> torch.Tensor:
    if sample_mask is None:
        return scope_mask.bool()
    return scope_mask.bool() & sample_mask.bool()


def _masked_mean(
    values: torch.Tensor,
    sample_mask: torch.Tensor | None,
) -> tuple[torch.Tensor, bool]:
    if sample_mask is None:
        return values.mean(), values.numel() > 0
    active_values = values[sample_mask]
    if active_values.numel() == 0:
        return values.new_zeros(()), False
    return active_values.mean(), True
