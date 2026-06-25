# src/rl_fzerox/core/policy/auxiliary_state/actor_regularization/mixin.py
"""Policy mixin that composes actor regularization losses and metrics."""

from __future__ import annotations

import math

import torch
from sb3x.common.auxiliary_losses import PolicyAuxiliaryLoss
from stable_baselines3.common.type_aliases import PyTorchObs

from rl_fzerox.core.policy.auxiliary_state.actor_regularization.distributions import (
    _AxisDistributionStats,
    _categorical_lean_expected_signed_values,
    _continuous_action_log_std,
    _continuous_action_mode,
    _discrete_pitch_distribution_stats,
    _split_lean_expected_signed_values,
)
from rl_fzerox.core.policy.auxiliary_state.actor_regularization.losses import (
    _combined_mask,
    _masked_mean,
    _signed_balance_loss,
    _std_cap_loss,
)
from rl_fzerox.core.policy.auxiliary_state.actor_regularization.metrics import (
    _metric_mean,
    _pitch_sample_metrics,
    _signed_balance_metrics,
)
from rl_fzerox.core.policy.auxiliary_state.target_tensors import (
    optional_auxiliary_targets,
    require_auxiliary_targets,
)
from rl_fzerox.core.policy.auxiliary_state.targets import (
    resolve_auxiliary_state_target,
)


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

            aux_targets = optional_auxiliary_targets(obs)
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
        aux_targets = require_auxiliary_targets(obs)
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

        aux_targets = require_auxiliary_targets(obs)
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
