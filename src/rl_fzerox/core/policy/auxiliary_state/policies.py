# src/rl_fzerox/core/policy/auxiliary_state/policies.py
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TypeAlias, TypedDict, Unpack

import torch
from gymnasium import spaces
from sb3_contrib.common.recurrent.type_aliases import RNNStates
from sb3x.common.auxiliary_losses import (
    PolicyActionEvaluation,
)
from sb3x.common.hybrid_action import ContinuousLogStdMode
from sb3x.common.maskable import MaybeMasks
from sb3x.common.recurrent import (
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
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule
from torch import nn
from torch.optim import Optimizer

from fzerox_emulator.arrays import BoolArray, NumpyArray, PolicyState
from rl_fzerox.core.envs.observations import ObservationValue
from rl_fzerox.core.policy.auxiliary_state.mixin import _AuxiliaryStatePolicyMixin
from rl_fzerox.core.policy.auxiliary_state.recurrent import _recurrent_tensor_state
from rl_fzerox.core.policy.auxiliary_state.targets import AuxiliaryStateTargetName

__all__ = [
    "AuxiliaryStateMaskableHybridActionMultiInputPolicy",
    "AuxiliaryStateMaskableHybridRecurrentMultiInputPolicy",
]


_Sb3Observation: TypeAlias = NumpyArray | dict[str, NumpyArray]


class _HybridPolicyKwargs(TypedDict, total=False):
    net_arch: list[int] | dict[str, list[int]] | None
    activation_fn: type[nn.Module]
    ortho_init: bool
    use_sde: bool
    log_std_init: float
    full_std: bool
    use_expln: bool
    squash_output: bool
    features_extractor_class: type[BaseFeaturesExtractor]
    features_extractor_kwargs: dict[str, object] | None
    share_features_extractor: bool
    normalize_images: bool
    optimizer_class: type[Optimizer]
    optimizer_kwargs: dict[str, object] | None
    hybrid_action_space: spaces.Dict | None
    hybrid_action_group_names: Mapping[str, Sequence[str]] | None
    continuous_log_std_mode: ContinuousLogStdMode
    continuous_log_std_bounds: tuple[float, float]


class _HybridRecurrentPolicyKwargs(_HybridPolicyKwargs, total=False):
    lstm_hidden_size: int
    n_lstm_layers: int
    shared_lstm: bool
    enable_critic_lstm: bool
    lstm_kwargs: dict[str, object] | None


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
        **kwargs: Unpack[_HybridPolicyKwargs],
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
        obs_tensor, _ = self.obs_to_tensor(_sb3_observation(observation))
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
        **kwargs: Unpack[_HybridRecurrentPolicyKwargs],
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
        obs_tensor, _ = self.obs_to_tensor(_sb3_observation(observation))
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


def _sb3_observation(observation: ObservationValue) -> _Sb3Observation:
    """Convert project observation values to the observation shape SB3 accepts."""

    if not isinstance(observation, dict):
        return observation
    converted: dict[str, NumpyArray] = {
        "image": observation["image"],
        "state": observation["state"],
    }
    auxiliary_targets = observation.get("auxiliary_state_targets")
    if auxiliary_targets is not None:
        converted["auxiliary_state_targets"] = auxiliary_targets
    return converted
