# src/rl_fzerox/core/policy/auxiliary_state/actor_regularization.py
"""Actor-side auxiliary regularization losses and metrics for policy training."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import torch
from sb3x.common.auxiliary_losses import PolicyAuxiliaryLoss
from stable_baselines3.common.type_aliases import PyTorchObs

from rl_fzerox.core.policy.auxiliary_state.target_tensors import (
    _optional_auxiliary_targets,
    _require_auxiliary_targets,
)
from rl_fzerox.core.policy.auxiliary_state.targets import (
    resolve_auxiliary_state_target,
)


@dataclass(frozen=True, slots=True)
class _AxisDistributionStats:
    mean: torch.Tensor
    std: torch.Tensor
    entropy: torch.Tensor
    source: Literal["continuous", "discrete"]
    log_std: torch.Tensor | None = None


@dataclass(frozen=True, slots=True)
class _SignedBalanceLoss:
    bias: torch.Tensor
    loss_value: torch.Tensor
    total_loss: torch.Tensor


class _ActorRegularizationMixin:
    _grounded_pitch_neutral_loss_weight: float
    _pitch_std_cap_loss_weight: float
    _grounded_pitch_std_cap: float
    _airborne_pitch_std_cap: float
    _steer_std_cap_loss_weight: float
    _steer_std_cap: float
    _steer_signed_balance_loss_weight: float
    _steer_signed_balance_deadzone: float
    _lean_signed_balance_loss_weight: float
    _lean_signed_balance_deadzone: float
    _continuous_steer_index: int | None
    _continuous_pitch_index: int | None
    _discrete_pitch_index: int | None
    _discrete_lean_index: int | None
    _discrete_lean_left_index: int | None
    _discrete_lean_right_index: int | None
    _continuous_action_group_count: int
    _pitch_bucket_values: tuple[float, ...]

    def _actor_regularization_loss(
        self,
        distribution: object,
        *,
        actions: torch.Tensor,
        obs: PyTorchObs,
        sample_mask: torch.Tensor | None = None,
    ) -> PolicyAuxiliaryLoss | None:
        if not self._actor_regularization_enabled():
            return None

        total_loss: torch.Tensor | None = None
        loss_anchor: torch.Tensor | None = None
        metrics: dict[str, float] = {}

        def add_loss(loss_value: torch.Tensor) -> None:
            nonlocal total_loss
            total_loss = loss_value if total_loss is None else total_loss + loss_value

        if self._pitch_actor_regularization_enabled():
            pitch_stats = self._pitch_distribution_stats(distribution)
            pitch_mean = pitch_stats.mean
            loss_anchor = pitch_mean

            mean_loss = self._grounded_pitch_mean_loss(
                pitch_mean,
                obs=obs,
                sample_mask=sample_mask,
            )
            if mean_loss is not None:
                loss_value, weighted_loss = mean_loss
                add_loss(weighted_loss)
                loss = float(loss_value.detach().cpu().item())
                weighted = float(weighted_loss.detach().cpu().item())
                metrics.update(
                    {
                        "actor/grounded_pitch_mean_loss": loss,
                        "actor/grounded_pitch_mean_loss_weighted": weighted,
                        "actor/grounded_pitch_neutral": loss,
                        "actor/grounded_pitch_neutral_weighted": weighted,
                    }
                )

            std_loss = self._pitch_std_cap_loss(
                pitch_stats,
                obs=obs,
                sample_mask=sample_mask,
            )
            if std_loss is not None:
                add_loss(std_loss.total_loss)
                metrics.update(std_loss.metrics)

            aux_targets = _optional_auxiliary_targets(obs)
            if aux_targets is not None:
                pitch_sample = self._pitch_action_sample(actions, reference=pitch_mean)
                metrics.update(
                    _pitch_sample_metrics(
                        pitch_mean=pitch_mean,
                        pitch_sample=pitch_sample,
                        aux_targets=aux_targets,
                        sample_mask=sample_mask,
                    )
                )

        if self._steer_actor_regularization_enabled():
            steer_index = self._continuous_steer_index
            if steer_index is None:
                raise RuntimeError("steer actor regularization was not initialized")
            steer_stats = self._continuous_axis_distribution_stats(
                distribution,
                axis_index=steer_index,
            )
            loss_anchor = steer_stats.std
            steer_loss = self._steer_std_cap_loss(
                steer_stats,
                sample_mask=sample_mask,
            )
            if steer_loss is not None:
                add_loss(steer_loss.total_loss)
                metrics.update(steer_loss.metrics)

            steer_balance_loss = _signed_balance_loss(
                steer_stats.mean,
                deadzone=self._steer_signed_balance_deadzone,
                loss_weight=self._steer_signed_balance_loss_weight,
                sample_mask=sample_mask,
            )
            if steer_balance_loss is not None:
                add_loss(steer_balance_loss.total_loss)
                metrics.update(_signed_balance_metrics("steer", steer_balance_loss))

        if self._lean_actor_regularization_enabled():
            lean_expected = self._lean_expected_signed_values(distribution)
            loss_anchor = lean_expected
            lean_balance_loss = _signed_balance_loss(
                lean_expected,
                deadzone=self._lean_signed_balance_deadzone,
                loss_weight=self._lean_signed_balance_loss_weight,
                sample_mask=sample_mask,
            )
            if lean_balance_loss is not None:
                add_loss(lean_balance_loss.total_loss)
                metrics.update(_signed_balance_metrics("lean", lean_balance_loss))

        if not metrics:
            return None
        if total_loss is None:
            if loss_anchor is None:
                return None
            total_loss = loss_anchor.new_zeros(())
        return PolicyAuxiliaryLoss(total_loss=total_loss, metrics=metrics)

    def _grounded_pitch_mean_loss(
        self,
        pitch_mean: torch.Tensor,
        *,
        obs: PyTorchObs,
        sample_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        if self._grounded_pitch_neutral_loss_weight <= 0.0:
            return None
        aux_targets = _require_auxiliary_targets(obs)
        airborne_index = resolve_auxiliary_state_target("vehicle_state.airborne").vector_start
        grounded = aux_targets[:, airborne_index] < 0.5
        loss_value, has_active_samples = _masked_mean(
            pitch_mean.square(),
            _combined_mask(grounded, sample_mask),
        )
        if not has_active_samples:
            return None
        return (
            loss_value,
            self._grounded_pitch_neutral_loss_weight * loss_value,
        )

    def _pitch_distribution_stats(self, distribution: object) -> _AxisDistributionStats:
        continuous_pitch_index = self._continuous_pitch_index
        if continuous_pitch_index is not None:
            return self._continuous_axis_distribution_stats(
                distribution,
                axis_index=continuous_pitch_index,
            )

        discrete_pitch_index = self._discrete_pitch_index
        if discrete_pitch_index is not None:
            return _discrete_pitch_distribution_stats(
                distribution,
                discrete_pitch_index=discrete_pitch_index,
                bucket_values=self._pitch_bucket_values,
            )

        raise RuntimeError("pitch action group was not initialized")

    def _pitch_action_sample(
        self,
        actions: torch.Tensor,
        *,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        continuous_pitch_index = self._continuous_pitch_index
        if continuous_pitch_index is not None:
            return actions[:, continuous_pitch_index].to(dtype=reference.dtype)

        discrete_pitch_index = self._discrete_pitch_index
        if discrete_pitch_index is None:
            raise RuntimeError("pitch action group was not initialized")
        action_index = self._continuous_action_group_count + discrete_pitch_index
        pitch_bucket_indices = actions[:, action_index].long()
        bucket_values = actions.new_tensor(self._pitch_bucket_values, dtype=reference.dtype)
        return bucket_values[pitch_bucket_indices]

    def _pitch_std_cap_loss(
        self,
        pitch_stats: _AxisDistributionStats,
        *,
        obs: PyTorchObs,
        sample_mask: torch.Tensor | None,
    ) -> PolicyAuxiliaryLoss | None:
        if self._pitch_std_cap_loss_weight <= 0.0:
            return None
        pitch_std = pitch_stats.std
        total_loss = pitch_std.new_zeros(())
        metrics: dict[str, float] = {
            "pitch/std": _metric_mean(pitch_std, sample_mask),
            "pitch/entropy": _metric_mean(pitch_stats.entropy, sample_mask),
        }
        if pitch_stats.log_std is not None:
            metrics["pitch/log_std"] = _metric_mean(pitch_stats.log_std, sample_mask)

        aux_targets = _require_auxiliary_targets(obs)
        airborne_index = resolve_auxiliary_state_target("vehicle_state.airborne").vector_start
        airborne = aux_targets[:, airborne_index] >= 0.5
        scoped_losses = (
            (
                "grounded",
                ~airborne,
                self._grounded_pitch_std_cap,
            ),
            (
                "airborne",
                airborne,
                self._airborne_pitch_std_cap,
            ),
        )
        for scope, scope_mask, cap in scoped_losses:
            metrics[f"pitch/{scope}_std"] = _metric_mean(
                pitch_std,
                _combined_mask(scope_mask, sample_mask),
            )
            if pitch_stats.source == "discrete" and scope == "airborne":
                continue
            loss_value = _std_cap_loss(
                pitch_std,
                cap=cap,
                sample_mask=_combined_mask(scope_mask, sample_mask),
            )
            if loss_value is None:
                continue
            weighted_loss = self._pitch_std_cap_loss_weight * loss_value
            total_loss = total_loss + weighted_loss
            metrics.update(
                {
                    f"pitch/{scope}_std_cap_loss": float(loss_value.detach().cpu().item()),
                    f"pitch/{scope}_std_cap_loss_weighted": float(
                        weighted_loss.detach().cpu().item()
                    ),
                }
            )

        return PolicyAuxiliaryLoss(total_loss=total_loss, metrics=metrics)

    def _steer_std_cap_loss(
        self,
        steer_stats: _AxisDistributionStats,
        *,
        sample_mask: torch.Tensor | None,
    ) -> PolicyAuxiliaryLoss | None:
        if self._steer_std_cap_loss_weight <= 0.0:
            return None
        steer_std = steer_stats.std
        total_loss = steer_std.new_zeros(())
        metrics: dict[str, float] = {
            "steer/std": _metric_mean(steer_std, sample_mask),
            "steer/entropy": _metric_mean(steer_stats.entropy, sample_mask),
        }
        if steer_stats.log_std is not None:
            metrics["steer/log_std"] = _metric_mean(steer_stats.log_std, sample_mask)

        loss_value = _std_cap_loss(
            steer_std,
            cap=self._steer_std_cap,
            sample_mask=sample_mask,
        )
        if loss_value is not None:
            weighted_loss = self._steer_std_cap_loss_weight * loss_value
            total_loss = total_loss + weighted_loss
            metrics.update(
                {
                    "steer/std_cap_loss": float(loss_value.detach().cpu().item()),
                    "steer/std_cap_loss_weighted": float(weighted_loss.detach().cpu().item()),
                }
            )

        return PolicyAuxiliaryLoss(total_loss=total_loss, metrics=metrics)

    def _lean_expected_signed_values(self, distribution: object) -> torch.Tensor:
        lean_index = self._discrete_lean_index
        if lean_index is not None:
            return _categorical_lean_expected_signed_values(
                distribution,
                branch_index=lean_index,
            )

        lean_left_index = self._discrete_lean_left_index
        lean_right_index = self._discrete_lean_right_index
        if lean_left_index is None or lean_right_index is None:
            raise RuntimeError("lean actor regularization was not initialized")
        return _split_lean_expected_signed_values(
            distribution,
            left_branch_index=lean_left_index,
            right_branch_index=lean_right_index,
        )

    def _continuous_axis_distribution_stats(
        self,
        distribution: object,
        *,
        axis_index: int,
    ) -> _AxisDistributionStats:
        continuous_mode = _continuous_action_mode(distribution)
        axis_mean = continuous_mode[:, axis_index]
        continuous_log_std = _continuous_action_log_std(distribution)
        axis_log_std = continuous_log_std[:, axis_index]
        axis_std = axis_log_std.exp()
        entropy = axis_log_std + 0.5 * math.log(2.0 * math.pi * math.e)
        return _AxisDistributionStats(
            mean=axis_mean,
            std=axis_std,
            entropy=entropy,
            source="continuous",
            log_std=axis_log_std,
        )

    def _actor_regularization_enabled(self) -> bool:
        return (
            self._pitch_actor_regularization_enabled()
            or self._steer_actor_regularization_enabled()
            or self._lean_actor_regularization_enabled()
        )

    def _pitch_actor_regularization_enabled(self) -> bool:
        return (
            self._grounded_pitch_neutral_loss_weight > 0.0 or self._pitch_std_cap_loss_weight > 0.0
        )

    def _steer_actor_regularization_enabled(self) -> bool:
        return self._steer_std_cap_loss_weight > 0.0 or self._steer_signed_balance_loss_weight > 0.0

    def _lean_actor_regularization_enabled(self) -> bool:
        return self._lean_signed_balance_loss_weight > 0.0


def _continuous_action_mode(distribution: object) -> torch.Tensor:
    continuous_dist = getattr(distribution, "continuous_dist", None)
    mode = getattr(continuous_dist, "mode", None)
    if not callable(mode):
        raise TypeError("Actor regularization requires a hybrid continuous distribution")
    value = mode()
    if not isinstance(value, torch.Tensor):
        raise TypeError("Continuous distribution mode must return a tensor")
    return value


def _continuous_action_log_std(distribution: object) -> torch.Tensor:
    continuous_log_std = getattr(distribution, "continuous_log_std", None)
    if not callable(continuous_log_std):
        raise TypeError("Actor regularization requires hybrid continuous log std access")
    value = continuous_log_std()
    if not isinstance(value, torch.Tensor):
        raise TypeError("Continuous distribution log std must return a tensor")
    log_std: torch.Tensor = value
    if log_std.ndim != 2:
        raise TypeError("Continuous distribution log std must be batched")
    return log_std


def _discrete_pitch_distribution_stats(
    distribution: object,
    *,
    discrete_pitch_index: int,
    bucket_values: tuple[float, ...],
) -> _AxisDistributionStats:
    branch_distribution = _discrete_branch_distribution(
        distribution,
        branch_index=discrete_pitch_index,
    )
    raw_probs = getattr(branch_distribution, "probs", None)
    if not isinstance(raw_probs, torch.Tensor):
        raise TypeError("Discrete pitch distribution must expose categorical probabilities")
    probs: torch.Tensor = raw_probs
    if probs.ndim != 2:
        raise TypeError("Discrete pitch probabilities must be batched")
    if probs.shape[1] != len(bucket_values):
        raise ValueError("Discrete pitch bucket count does not match the pitch distribution shape")

    bucket_tensor = probs.new_tensor(bucket_values).unsqueeze(0)
    mean = (probs * bucket_tensor).sum(dim=1)
    variance = (probs * (bucket_tensor - mean.unsqueeze(1)).square()).sum(dim=1)
    std = variance.clamp(min=0.0).sqrt()

    entropy_fn = getattr(branch_distribution, "entropy", None)
    if not callable(entropy_fn):
        raise TypeError("Discrete pitch distribution must expose entropy")
    entropy = entropy_fn()
    if not isinstance(entropy, torch.Tensor):
        raise TypeError("Discrete pitch entropy must be a tensor")
    return _AxisDistributionStats(mean=mean, std=std, entropy=entropy, source="discrete")


def _categorical_lean_expected_signed_values(
    distribution: object,
    *,
    branch_index: int,
) -> torch.Tensor:
    probs = _discrete_branch_probabilities(
        distribution,
        branch_index=branch_index,
        label="categorical lean",
    )
    if probs.shape[1] not in {3, 4}:
        raise ValueError("Categorical lean probabilities must have three or four values")
    signed_values = probs.new_zeros((probs.shape[1],))
    signed_values[1] = -1.0
    signed_values[2] = 1.0
    return (probs * signed_values.unsqueeze(0)).sum(dim=1)


def _split_lean_expected_signed_values(
    distribution: object,
    *,
    left_branch_index: int,
    right_branch_index: int,
) -> torch.Tensor:
    left_probs = _discrete_branch_probabilities(
        distribution,
        branch_index=left_branch_index,
        label="lean_left",
    )
    right_probs = _discrete_branch_probabilities(
        distribution,
        branch_index=right_branch_index,
        label="lean_right",
    )
    if left_probs.shape[1] != 2 or right_probs.shape[1] != 2:
        raise ValueError("Split lean probabilities must be binary")
    return right_probs[:, 1] - left_probs[:, 1]


def _discrete_branch_probabilities(
    distribution: object,
    *,
    branch_index: int,
    label: str,
) -> torch.Tensor:
    branch_distribution = _discrete_branch_distribution(
        distribution,
        branch_index=branch_index,
    )
    raw_probs = getattr(branch_distribution, "probs", None)
    if not isinstance(raw_probs, torch.Tensor):
        raise TypeError(f"Discrete {label} distribution must expose categorical probabilities")
    probs: torch.Tensor = raw_probs
    if probs.ndim != 2:
        raise TypeError(f"Discrete {label} probabilities must be batched")
    return probs


def _discrete_branch_distribution(
    distribution: object,
    *,
    branch_index: int,
) -> object:
    discrete_dist = getattr(distribution, "discrete_dist", None)
    branches = getattr(discrete_dist, "distributions", None)
    if not isinstance(branches, Sequence):
        raise TypeError("Actor regularization requires a hybrid discrete distribution")
    try:
        return branches[branch_index]
    except IndexError as exc:
        raise ValueError("Discrete branch index is outside the distribution") from exc


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
