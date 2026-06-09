# src/rl_fzerox/core/policy/auxiliary_state/policies.py
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from gymnasium import spaces
from sb3_contrib.common.recurrent.type_aliases import RNNStates
from sb3x.common.auxiliary_losses import (
    PolicyActionEvaluation,
    PolicyAuxiliaryLoss,
    combine_policy_auxiliary_losses,
)
from sb3x.common.maskable import MaybeMasks
from sb3x.common.recurrent import (
    count_vectorized_envs,
    require_linear,
    require_lstm_state,
    split_actor_critic_features,
)
from sb3x.ppo_mask_hybrid_action.policies import (
    MaskableHybridActionMultiInputActorCriticPolicy,
)
from sb3x.ppo_mask_hybrid_recurrent.policies import (
    MaskableHybridRecurrentMultiInputActorCriticPolicy,
)
from sb3x.ppo_mask_recurrent.policies import MaskableRecurrentMultiInputActorCriticPolicy
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule
from torch import nn

from fzerox_emulator.arrays import BoolArray, PolicyState
from rl_fzerox.core.envs.observations import ObservationValue
from rl_fzerox.core.policy.auxiliary_state.heads import (
    AuxiliaryStateHeadBank,
    AuxiliaryStateLossTerm,
)
from rl_fzerox.core.policy.auxiliary_state.observations import (
    auxiliary_state_targets_field,
)
from rl_fzerox.core.policy.auxiliary_state.targets import (
    AuxiliaryStateTargetName,
    is_auxiliary_state_target_name,
    resolve_auxiliary_state_target,
)


@dataclass(frozen=True, slots=True)
class _AxisDistributionStats:
    mean: torch.Tensor
    std: torch.Tensor
    entropy: torch.Tensor
    source: Literal["continuous", "discrete"]
    log_std: torch.Tensor | None = None


def _auxiliary_state_loss_terms(
    config: Mapping[str, object] | None,
) -> tuple[AuxiliaryStateLossTerm, ...]:
    if config is None:
        return ()
    losses = config.get("losses")
    if not isinstance(losses, Sequence):
        return ()
    resolved: list[AuxiliaryStateLossTerm] = []
    for entry in losses:
        if not isinstance(entry, Mapping):
            raise TypeError("policy auxiliary loss entries must be mappings")
        name = entry.get("name")
        weight = entry.get("weight", 1.0)
        grounded_only = entry.get("grounded_only", False)
        if not isinstance(name, str):
            raise TypeError("policy auxiliary loss names must be strings")
        if not is_auxiliary_state_target_name(name):
            raise ValueError(f"Unsupported policy auxiliary loss target: {name!r}")
        target = resolve_auxiliary_state_target(name)
        resolved.append(
            AuxiliaryStateLossTerm(
                name=target.name,
                weight=float(weight),
                grounded_only=bool(grounded_only),
            )
        )
    return tuple(resolved)


def _auxiliary_head_arch(config: Mapping[str, object] | None) -> tuple[int, ...]:
    if config is None:
        return ()
    raw_head_arch = config.get("head_arch")
    if not isinstance(raw_head_arch, Sequence):
        return ()
    resolved: list[int] = []
    for value in raw_head_arch:
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise TypeError("policy auxiliary head widths must be numeric")
        width = int(value)
        if width <= 0:
            raise ValueError("policy auxiliary head widths must be positive")
        resolved.append(width)
    return tuple(resolved)


def _require_auxiliary_targets(obs: PyTorchObs) -> torch.Tensor:
    if not isinstance(obs, Mapping):
        raise TypeError("Auxiliary-state policies require dict observations")
    field_name = auxiliary_state_targets_field()
    aux_targets = obs.get(field_name)
    if not isinstance(aux_targets, torch.Tensor):
        raise TypeError(f"Auxiliary-state policies require tensor observation key {field_name!r}")
    aux_target_tensor: torch.Tensor = aux_targets
    return torch.flatten(aux_target_tensor.float(), start_dim=1)


class _AuxiliaryStatePolicyMixin:
    _auxiliary_state_losses: tuple[AuxiliaryStateLossTerm, ...]
    _auxiliary_state_heads: AuxiliaryStateHeadBank | None
    _grounded_pitch_neutral_loss_weight: float
    _pitch_std_cap_loss_weight: float
    _grounded_pitch_std_cap: float
    _airborne_pitch_std_cap: float
    _steer_std_cap_loss_weight: float
    _steer_std_cap: float
    _continuous_steer_index: int | None
    _continuous_pitch_index: int | None
    _discrete_pitch_index: int | None
    _continuous_action_group_count: int
    _pitch_bucket_values: tuple[float, ...]

    def _init_auxiliary_state(
        self,
        *,
        input_dim: int,
        auxiliary_state: Mapping[str, object] | None,
        actor_regularization: Mapping[str, object] | None,
        continuous_action_group_names: Sequence[str] = (),
        discrete_action_group_names: Sequence[str] = (),
        pitch_bucket_count: int = 5,
        activation_fn: type[nn.Module],
    ) -> None:
        self._auxiliary_state_losses = _auxiliary_state_loss_terms(auxiliary_state)
        self._auxiliary_state_heads = (
            AuxiliaryStateHeadBank(
                input_dim=input_dim,
                head_arch=_auxiliary_head_arch(auxiliary_state),
                activation_fn=activation_fn,
            )
            if auxiliary_state is not None
            else None
        )
        self._grounded_pitch_neutral_loss_weight = _grounded_pitch_neutral_loss_weight(
            actor_regularization
        )
        self._pitch_std_cap_loss_weight = _pitch_std_cap_loss_weight(actor_regularization)
        self._grounded_pitch_std_cap = _grounded_pitch_std_cap(actor_regularization)
        self._airborne_pitch_std_cap = _airborne_pitch_std_cap(actor_regularization)
        self._steer_std_cap_loss_weight = _steer_std_cap_loss_weight(actor_regularization)
        self._steer_std_cap = _steer_std_cap(actor_regularization)
        continuous_names = tuple(continuous_action_group_names)
        discrete_names = tuple(discrete_action_group_names)
        self._continuous_steer_index = _axis_index(continuous_names, "steer")
        self._continuous_pitch_index = _axis_index(continuous_names, "pitch")
        self._discrete_pitch_index = _axis_index(discrete_names, "pitch")
        self._continuous_action_group_count = len(continuous_names)
        self._pitch_bucket_values = _pitch_bucket_values(pitch_bucket_count)
        if (
            self._pitch_actor_regularization_enabled()
            and self._continuous_pitch_index is None
            and self._discrete_pitch_index is None
        ):
            raise ValueError("pitch actor regularization requires a pitch action group")
        if self._steer_actor_regularization_enabled() and self._continuous_steer_index is None:
            raise ValueError("steer actor regularization requires a continuous steer action group")

    def _auxiliary_state_loss(
        self,
        latent: torch.Tensor,
        *,
        obs: PyTorchObs,
        sample_mask: torch.Tensor | None = None,
    ) -> PolicyAuxiliaryLoss | None:
        if self._auxiliary_state_heads is None:
            return None
        return self._auxiliary_state_heads.loss(
            latent,
            aux_targets=_require_auxiliary_targets(obs),
            losses=self._auxiliary_state_losses,
            sample_mask=sample_mask,
        )

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
                pitch_sample = self._pitch_action_sample(actions, dtype=pitch_mean.dtype)
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
        dtype: torch.dtype,
    ) -> torch.Tensor:
        continuous_pitch_index = self._continuous_pitch_index
        if continuous_pitch_index is not None:
            return actions[:, continuous_pitch_index].to(dtype=dtype)

        discrete_pitch_index = self._discrete_pitch_index
        if discrete_pitch_index is None:
            raise RuntimeError("pitch action group was not initialized")
        action_index = self._continuous_action_group_count + discrete_pitch_index
        pitch_bucket_indices = actions[:, action_index].long()
        bucket_values = actions.new_tensor(self._pitch_bucket_values, dtype=dtype)
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
            self._pitch_actor_regularization_enabled() or self._steer_actor_regularization_enabled()
        )

    def _pitch_actor_regularization_enabled(self) -> bool:
        return (
            self._grounded_pitch_neutral_loss_weight > 0.0 or self._pitch_std_cap_loss_weight > 0.0
        )

    def _steer_actor_regularization_enabled(self) -> bool:
        return self._steer_std_cap_loss_weight > 0.0

    def _combined_policy_auxiliary_loss(
        self,
        *,
        source_latent: torch.Tensor,
        distribution: object,
        actions: torch.Tensor,
        obs: PyTorchObs,
        sample_mask: torch.Tensor | None = None,
    ) -> PolicyAuxiliaryLoss | None:
        return combine_policy_auxiliary_losses(
            (
                self._auxiliary_state_loss(
                    source_latent,
                    obs=obs,
                    sample_mask=sample_mask,
                ),
                self._actor_regularization_loss(
                    distribution,
                    actions=actions,
                    obs=obs,
                    sample_mask=sample_mask,
                ),
            )
        )

    def _auxiliary_state_predictions(
        self,
        latent: torch.Tensor,
        *,
        names: Sequence[AuxiliaryStateTargetName] | None = None,
    ) -> dict[str, object]:
        if self._auxiliary_state_heads is None:
            raise RuntimeError("Policy does not have auxiliary-state prediction heads")
        predictions = self._auxiliary_state_heads.predict_values(latent, names=names)
        return {str(name): value for name, value in predictions.items()}


def _grounded_pitch_neutral_loss_weight(
    config: Mapping[str, object] | None,
) -> float:
    return _non_negative_config_float(
        config,
        key="grounded_pitch_neutral_loss_weight",
        label="grounded pitch neutral loss weight",
        default=0.0,
    )


def _pitch_std_cap_loss_weight(
    config: Mapping[str, object] | None,
) -> float:
    return _non_negative_config_float(
        config,
        key="pitch_std_cap_loss_weight",
        label="pitch std cap loss weight",
        default=0.0,
    )


def _grounded_pitch_std_cap(
    config: Mapping[str, object] | None,
) -> float:
    return _positive_config_float(
        config,
        key="grounded_pitch_std_cap",
        label="grounded pitch std cap",
        default=0.35,
    )


def _airborne_pitch_std_cap(
    config: Mapping[str, object] | None,
) -> float:
    return _positive_config_float(
        config,
        key="airborne_pitch_std_cap",
        label="airborne pitch std cap",
        default=0.8,
    )


def _steer_std_cap_loss_weight(
    config: Mapping[str, object] | None,
) -> float:
    return _non_negative_config_float(
        config,
        key="steer_std_cap_loss_weight",
        label="steer std cap loss weight",
        default=0.0,
    )


def _steer_std_cap(
    config: Mapping[str, object] | None,
) -> float:
    return _positive_config_float(
        config,
        key="steer_std_cap",
        label="steer std cap",
        default=1.0,
    )


def _non_negative_config_float(
    config: Mapping[str, object] | None,
    *,
    key: str,
    label: str,
    default: float,
) -> float:
    if config is None:
        return default
    value = config.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{label} must be numeric")
    resolved = float(value)
    if resolved < 0.0:
        raise ValueError(f"{label} must be non-negative")
    return resolved


def _positive_config_float(
    config: Mapping[str, object] | None,
    *,
    key: str,
    label: str,
    default: float,
) -> float:
    if config is None:
        return default
    value = config.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{label} must be numeric")
    resolved = float(value)
    if resolved <= 0.0:
        raise ValueError(f"{label} must be positive")
    return resolved


def _axis_index(names: Sequence[str], axis: str) -> int | None:
    try:
        return tuple(names).index(axis)
    except ValueError:
        return None


def _pitch_bucket_values(bucket_count: int) -> tuple[float, ...]:
    neutral_index = int(bucket_count) // 2
    if neutral_index <= 0:
        return (0.0,)
    return tuple(
        float(index - neutral_index) / float(neutral_index) for index in range(int(bucket_count))
    )


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
    std = torch.sqrt(torch.clamp(variance, min=0.0))

    entropy_fn = getattr(branch_distribution, "entropy", None)
    if not callable(entropy_fn):
        raise TypeError("Discrete pitch distribution must expose entropy")
    entropy = entropy_fn()
    if not isinstance(entropy, torch.Tensor):
        raise TypeError("Discrete pitch entropy must be a tensor")
    return _AxisDistributionStats(mean=mean, std=std, entropy=entropy, source="discrete")


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
        raise ValueError("Discrete pitch branch index is outside the distribution") from exc


def _optional_auxiliary_targets(obs: PyTorchObs) -> torch.Tensor | None:
    if not isinstance(obs, Mapping):
        return None
    field_name = auxiliary_state_targets_field()
    aux_targets = obs.get(field_name)
    if not isinstance(aux_targets, torch.Tensor):
        return None
    aux_target_tensor: torch.Tensor = aux_targets
    return torch.flatten(aux_target_tensor.float(), start_dim=1)


def _pitch_sample_metrics(
    *,
    pitch_mean: torch.Tensor,
    pitch_sample: torch.Tensor,
    aux_targets: torch.Tensor,
    sample_mask: torch.Tensor | None,
) -> dict[str, float]:
    airborne_index = resolve_auxiliary_state_target("vehicle_state.airborne").vector_start
    airborne = (aux_targets[:, airborne_index] >= 0.5).to(dtype=torch.bool)
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
    per_sample = torch.relu(values - values.new_tensor(cap)).square()
    loss_value, has_active_samples = _masked_mean(per_sample, sample_mask)
    if not has_active_samples:
        return None
    return loss_value


def _combined_mask(
    scope_mask: torch.Tensor,
    sample_mask: torch.Tensor | None,
) -> torch.Tensor:
    if sample_mask is None:
        return scope_mask.to(dtype=torch.bool)
    return scope_mask.to(dtype=torch.bool) & sample_mask.to(dtype=torch.bool)


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


def _recurrent_tensor_state(
    *,
    policy: MaskableRecurrentMultiInputActorCriticPolicy
    | MaskableHybridRecurrentMultiInputActorCriticPolicy,
    state: PolicyState,
    obs: PyTorchObs,
    episode_start: BoolArray | None,
) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
    n_envs = count_vectorized_envs(obs)
    if state is None:
        zeros = np.concatenate(
            [np.zeros(policy.lstm_hidden_state_shape) for _ in range(n_envs)],
            axis=1,
        )
        state = (zeros, zeros)
    if episode_start is None:
        episode_start = np.zeros(n_envs, dtype=bool)

    return (
        (
            torch.tensor(state[0], dtype=torch.float32, device=policy.device),
            torch.tensor(state[1], dtype=torch.float32, device=policy.device),
        ),
        torch.tensor(
            episode_start,
            dtype=torch.float32,
            device=policy.device,
        ),
    )


class AuxiliaryStateMaskableHybridActionMultiInputPolicy(
    _AuxiliaryStatePolicyMixin,
    MaskableHybridActionMultiInputActorCriticPolicy,
):
    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space: spaces.Box,
        lr_schedule: Schedule,
        *,
        auxiliary_state: Mapping[str, object] | None = None,
        actor_regularization: Mapping[str, object] | None = None,
        continuous_action_group_names: Sequence[str] = (),
        discrete_action_group_names: Sequence[str] = (),
        pitch_bucket_count: int = 5,
        **kwargs: object,
    ) -> None:
        MaskableHybridActionMultiInputActorCriticPolicy.__init__(
            self,
            observation_space,
            action_space,
            lr_schedule,
            **kwargs,
        )
        self._init_auxiliary_state(
            input_dim=int(self.features_dim),
            auxiliary_state=auxiliary_state,
            actor_regularization=actor_regularization,
            continuous_action_group_names=continuous_action_group_names,
            discrete_action_group_names=discrete_action_group_names,
            pitch_bucket_count=pitch_bucket_count,
            activation_fn=self.activation_fn,
        )

    def evaluate_actions_with_aux(
        self,
        obs: PyTorchObs,
        actions: torch.Tensor,
        *,
        action_masks: MaybeMasks = None,
        auxiliary_mask: torch.Tensor | None = None,
    ) -> PolicyActionEvaluation:
        features = self.extract_features(obs)
        if self.share_features_extractor:
            if not isinstance(features, torch.Tensor):
                raise TypeError("Expected shared feature extractor to return a tensor")
            source_latent = features
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            if not isinstance(features, tuple) or len(features) != 2:
                raise TypeError("Expected separate actor and critic feature tensors")
            pi_features, vf_features = features
            source_latent = pi_features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)

        distribution = self._get_action_dist_from_latent(latent_pi)
        distribution.apply_masking(action_masks)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        entropy = distribution.entropy()
        aux_loss = self._combined_policy_auxiliary_loss(
            source_latent=source_latent,
            distribution=distribution,
            actions=actions,
            obs=obs,
            sample_mask=auxiliary_mask,
        )
        return PolicyActionEvaluation(
            values=values,
            log_prob=log_prob,
            entropy=entropy,
            aux_loss=aux_loss,
            entropy_components=distribution.entropy_components() or {},
        )

    def predict_auxiliary_state(
        self,
        observation: ObservationValue,
        *,
        state: PolicyState = None,
        episode_start: BoolArray | None = None,
        target_names: Sequence[AuxiliaryStateTargetName] | None = None,
    ) -> dict[str, object]:
        del state, episode_start
        self.set_training_mode(False)
        obs_tensor, _ = self.obs_to_tensor(observation)
        with torch.no_grad():
            features = self.extract_features(obs_tensor)
            if self.share_features_extractor:
                if not isinstance(features, torch.Tensor):
                    raise TypeError("Expected shared feature extractor to return a tensor")
                source_latent = features
            else:
                if not isinstance(features, tuple) or len(features) != 2:
                    raise TypeError("Expected separate actor and critic feature tensors")
                source_latent = features[0]
        return self._auxiliary_state_predictions(source_latent, names=target_names)


class AuxiliaryStateMaskableRecurrentMultiInputPolicy(
    _AuxiliaryStatePolicyMixin,
    MaskableRecurrentMultiInputActorCriticPolicy,
):
    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        *,
        auxiliary_state: Mapping[str, object] | None = None,
        actor_regularization: Mapping[str, object] | None = None,
        continuous_action_group_names: Sequence[str] = (),
        discrete_action_group_names: Sequence[str] = (),
        pitch_bucket_count: int = 5,
        **kwargs: object,
    ) -> None:
        MaskableRecurrentMultiInputActorCriticPolicy.__init__(
            self,
            observation_space,
            action_space,
            lr_schedule,
            **kwargs,
        )
        self._init_auxiliary_state(
            input_dim=int(self.lstm_output_dim),
            auxiliary_state=auxiliary_state,
            actor_regularization=actor_regularization,
            continuous_action_group_names=continuous_action_group_names,
            discrete_action_group_names=discrete_action_group_names,
            pitch_bucket_count=pitch_bucket_count,
            activation_fn=self.activation_fn,
        )

    def evaluate_actions_with_aux(
        self,
        obs: PyTorchObs,
        actions: torch.Tensor,
        lstm_states: RNNStates,
        episode_starts: torch.Tensor,
        *,
        action_masks: MaybeMasks = None,
        auxiliary_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, PolicyAuxiliaryLoss | None]:
        features = self.extract_features(obs)
        pi_features, vf_features = split_actor_critic_features(
            features,
            share_features_extractor=self.share_features_extractor,
        )

        latent_pi_source, _ = self._process_sequence(
            pi_features,
            require_lstm_state(lstm_states.pi),
            episode_starts,
            self.lstm_actor,
        )
        if self.lstm_critic is not None:
            latent_vf_source, _ = self._process_sequence(
                vf_features,
                require_lstm_state(lstm_states.vf),
                episode_starts,
                self.lstm_critic,
            )
        elif self.shared_lstm:
            latent_vf_source = latent_pi_source.detach()
        else:
            latent_vf_source = require_linear(self.critic)(vf_features)

        latent_pi = self.mlp_extractor.forward_actor(latent_pi_source)
        latent_vf = self.mlp_extractor.forward_critic(latent_vf_source)
        distribution = self._get_action_dist_from_latent(latent_pi)
        if action_masks is not None:
            distribution.apply_masking(action_masks)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        entropy = distribution.entropy()
        aux_loss = self._auxiliary_state_loss(
            latent_pi_source,
            obs=obs,
            sample_mask=auxiliary_mask,
        )
        return values, log_prob, entropy, aux_loss

    def predict_auxiliary_state(
        self,
        observation: ObservationValue,
        *,
        state: PolicyState = None,
        episode_start: BoolArray | None = None,
        target_names: Sequence[AuxiliaryStateTargetName] | None = None,
    ) -> dict[str, object]:
        self.set_training_mode(False)
        obs_tensor, _ = self.obs_to_tensor(observation)
        tensor_states, tensor_episode_starts = _recurrent_tensor_state(
            policy=self,
            state=state,
            obs=obs_tensor,
            episode_start=episode_start,
        )
        with torch.no_grad():
            features = self.extract_features(obs_tensor)
            pi_features, _ = split_actor_critic_features(
                features,
                share_features_extractor=self.share_features_extractor,
            )
            latent_pi_source, _ = self._process_sequence(
                pi_features,
                tensor_states,
                tensor_episode_starts,
                self.lstm_actor,
            )
        return self._auxiliary_state_predictions(latent_pi_source, names=target_names)


class AuxiliaryStateMaskableHybridRecurrentMultiInputPolicy(
    _AuxiliaryStatePolicyMixin,
    MaskableHybridRecurrentMultiInputActorCriticPolicy,
):
    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space: spaces.Box,
        lr_schedule: Schedule,
        *,
        auxiliary_state: Mapping[str, object] | None = None,
        actor_regularization: Mapping[str, object] | None = None,
        continuous_action_group_names: Sequence[str] = (),
        discrete_action_group_names: Sequence[str] = (),
        pitch_bucket_count: int = 5,
        **kwargs: object,
    ) -> None:
        MaskableHybridRecurrentMultiInputActorCriticPolicy.__init__(
            self,
            observation_space,
            action_space,
            lr_schedule,
            **kwargs,
        )
        self._init_auxiliary_state(
            input_dim=int(self.lstm_output_dim),
            auxiliary_state=auxiliary_state,
            actor_regularization=actor_regularization,
            continuous_action_group_names=continuous_action_group_names,
            discrete_action_group_names=discrete_action_group_names,
            pitch_bucket_count=pitch_bucket_count,
            activation_fn=self.activation_fn,
        )

    def evaluate_actions_with_aux(
        self,
        obs: PyTorchObs,
        actions: torch.Tensor,
        lstm_states: RNNStates,
        episode_starts: torch.Tensor,
        *,
        action_masks: MaybeMasks = None,
        auxiliary_mask: torch.Tensor | None = None,
    ) -> PolicyActionEvaluation:
        features = self.extract_features(obs)
        pi_features, vf_features = split_actor_critic_features(
            features,
            share_features_extractor=self.share_features_extractor,
        )

        latent_pi_source, _ = self._process_sequence(
            pi_features,
            require_lstm_state(lstm_states.pi),
            episode_starts,
            self.lstm_actor,
        )
        if self.lstm_critic is not None:
            latent_vf_source, _ = self._process_sequence(
                vf_features,
                require_lstm_state(lstm_states.vf),
                episode_starts,
                self.lstm_critic,
            )
        elif self.shared_lstm:
            latent_vf_source = latent_pi_source.detach()
        else:
            latent_vf_source = require_linear(self.critic)(vf_features)

        latent_pi = self.mlp_extractor.forward_actor(latent_pi_source)
        latent_vf = self.mlp_extractor.forward_critic(latent_vf_source)
        distribution = self._get_action_dist_from_latent(latent_pi)
        distribution.apply_masking(action_masks)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        entropy = distribution.entropy()
        aux_loss = self._combined_policy_auxiliary_loss(
            source_latent=latent_pi_source,
            distribution=distribution,
            actions=actions,
            obs=obs,
            sample_mask=auxiliary_mask,
        )
        return PolicyActionEvaluation(
            values=values,
            log_prob=log_prob,
            entropy=entropy,
            aux_loss=aux_loss,
            entropy_components=distribution.entropy_components() or {},
        )

    def predict_auxiliary_state(
        self,
        observation: ObservationValue,
        *,
        state: PolicyState = None,
        episode_start: BoolArray | None = None,
        target_names: Sequence[AuxiliaryStateTargetName] | None = None,
    ) -> dict[str, object]:
        self.set_training_mode(False)
        obs_tensor, _ = self.obs_to_tensor(observation)
        tensor_states, tensor_episode_starts = _recurrent_tensor_state(
            policy=self,
            state=state,
            obs=obs_tensor,
            episode_start=episode_start,
        )
        with torch.no_grad():
            features = self.extract_features(obs_tensor)
            pi_features, _ = split_actor_critic_features(
                features,
                share_features_extractor=self.share_features_extractor,
            )
            latent_pi_source, _ = self._process_sequence(
                pi_features,
                tensor_states,
                tensor_episode_starts,
                self.lstm_actor,
            )
        return self._auxiliary_state_predictions(latent_pi_source, names=target_names)
