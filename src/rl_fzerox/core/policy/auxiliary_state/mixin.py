# src/rl_fzerox/core/policy/auxiliary_state/mixin.py
from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from sb3x.common.auxiliary_losses import (
    PolicyAuxiliaryLoss,
    combine_policy_auxiliary_losses,
)
from stable_baselines3.common.type_aliases import PyTorchObs
from torch import nn

from rl_fzerox.core.policy.auxiliary_state.actor_regularization import (
    _ActorRegularizationMixin,
)
from rl_fzerox.core.policy.auxiliary_state.config import (
    _airborne_pitch_std_cap,
    _auxiliary_head_arch,
    _auxiliary_state_loss_terms,
    _axis_index,
    _grounded_pitch_neutral_loss_weight,
    _grounded_pitch_std_cap,
    _lean_signed_balance_deadzone,
    _lean_signed_balance_loss_weight,
    _pitch_bucket_values,
    _pitch_std_cap_loss_weight,
    _steer_signed_balance_deadzone,
    _steer_signed_balance_loss_weight,
    _steer_std_cap,
    _steer_std_cap_loss_weight,
)
from rl_fzerox.core.policy.auxiliary_state.heads import (
    AuxiliaryStateHeadBank,
    AuxiliaryStateLossTerm,
)
from rl_fzerox.core.policy.auxiliary_state.target_tensors import (
    _require_auxiliary_targets,
)
from rl_fzerox.core.policy.auxiliary_state.targets import (
    AuxiliaryStateTargetName,
)


class _AuxiliaryStatePolicyMixin(_ActorRegularizationMixin):
    _auxiliary_state_losses: tuple[AuxiliaryStateLossTerm, ...]
    _auxiliary_state_heads: AuxiliaryStateHeadBank | None
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
        self._steer_signed_balance_loss_weight = _steer_signed_balance_loss_weight(
            actor_regularization
        )
        self._steer_signed_balance_deadzone = _steer_signed_balance_deadzone(actor_regularization)
        self._lean_signed_balance_loss_weight = _lean_signed_balance_loss_weight(
            actor_regularization
        )
        self._lean_signed_balance_deadzone = _lean_signed_balance_deadzone(actor_regularization)
        continuous_names = tuple(continuous_action_group_names)
        discrete_names = tuple(discrete_action_group_names)
        self._continuous_steer_index = _axis_index(continuous_names, "steer")
        self._continuous_pitch_index = _axis_index(continuous_names, "pitch")
        self._discrete_pitch_index = _axis_index(discrete_names, "pitch")
        self._discrete_lean_index = _axis_index(discrete_names, "lean")
        self._discrete_lean_left_index = _axis_index(discrete_names, "lean_left")
        self._discrete_lean_right_index = _axis_index(discrete_names, "lean_right")
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
        if self._lean_actor_regularization_enabled() and not (
            self._discrete_lean_index is not None
            or (
                self._discrete_lean_left_index is not None
                and self._discrete_lean_right_index is not None
            )
        ):
            raise ValueError("lean actor regularization requires a lean action group")

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
