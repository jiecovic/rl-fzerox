# src/rl_fzerox/core/policy/auxiliary_state/actor_regularization/metrics.py
"""Metric formatting helpers for actor regularization diagnostics.

This module converts tensor loss state and sampled pitch actions into scalar
logging values. Keeping key names here makes the mixin focus on loss flow.
"""

from __future__ import annotations

import torch

from rl_fzerox.core.policy.auxiliary_state.actor_regularization.losses import (
    _combined_mask,
    _masked_mean,
    _SignedBalanceLoss,
)
from rl_fzerox.core.policy.auxiliary_state.targets import (
    resolve_auxiliary_state_target,
)


def _signed_balance_metrics(prefix: str, loss: _SignedBalanceLoss) -> dict[str, float]:
    return {
        f"{prefix}/signed_bias": float(loss.bias.detach().cpu().item()),
        f"{prefix}/signed_balance_loss": float(loss.loss_value.detach().cpu().item()),
        f"{prefix}/signed_balance_loss_weighted": float(loss.total_loss.detach().cpu().item()),
    }


def _pitch_sample_metrics(
    *,
    pitch_mean: torch.Tensor,
    pitch_sample: torch.Tensor,
    aux_targets: torch.Tensor,
    sample_mask: torch.Tensor | None,
) -> dict[str, float]:
    airborne_index = resolve_auxiliary_state_target("vehicle_state.airborne").vector_start
    airborne = (aux_targets[:, airborne_index] >= 0.5).bool()
    grounded = ~airborne
    near_saturation = (pitch_sample.abs() > 0.95).to(dtype=pitch_mean.dtype)
    metrics: dict[str, float] = {
        "pitch/raw_sample_saturation_fraction": _metric_mean(near_saturation, sample_mask),
    }
    scoped_metrics = {
        "pitch/mean_ground_abs": (pitch_mean.abs(), grounded),
        "pitch/mean_air_abs": (pitch_mean.abs(), airborne),
        "pitch/raw_sample_ground_abs": (pitch_sample.abs(), grounded),
        "pitch/raw_sample_air_abs": (pitch_sample.abs(), airborne),
    }
    for name, (values, scope_mask) in scoped_metrics.items():
        combined_mask = _combined_mask(scope_mask, sample_mask)
        value, has_active_samples = _masked_mean(values, combined_mask)
        if has_active_samples:
            metrics[name] = float(value.detach().cpu().item())
    return metrics


def _metric_mean(values: torch.Tensor, sample_mask: torch.Tensor | None) -> float:
    value, has_active_samples = _masked_mean(values, sample_mask)
    if not has_active_samples:
        return 0.0
    return float(value.detach().cpu().item())
