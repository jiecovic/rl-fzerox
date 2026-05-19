from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import torch
from gymnasium import spaces
from sb3_contrib.common.recurrent.type_aliases import RNNStates
from sb3x.common.auxiliary_losses import PolicyAuxiliaryLoss
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
    return torch.flatten(aux_targets.float(), start_dim=1)


class _AuxiliaryStatePolicyMixin:
    _auxiliary_state_losses: tuple[AuxiliaryStateLossTerm, ...]
    _auxiliary_state_heads: AuxiliaryStateHeadBank

    def _init_auxiliary_state(
        self,
        *,
        input_dim: int,
        auxiliary_state: Mapping[str, object] | None,
        activation_fn: type[nn.Module],
    ) -> None:
        self._auxiliary_state_losses = _auxiliary_state_loss_terms(auxiliary_state)
        self._auxiliary_state_heads = AuxiliaryStateHeadBank(
            input_dim=input_dim,
            head_arch=_auxiliary_head_arch(auxiliary_state),
            activation_fn=activation_fn,
        )

    def _auxiliary_state_loss(
        self,
        latent: torch.Tensor,
        *,
        obs: PyTorchObs,
        sample_mask: torch.Tensor | None = None,
    ) -> PolicyAuxiliaryLoss | None:
        return self._auxiliary_state_heads.loss(
            latent,
            aux_targets=_require_auxiliary_targets(obs),
            losses=self._auxiliary_state_losses,
            sample_mask=sample_mask,
        )

    def _auxiliary_state_predictions(
        self,
        latent: torch.Tensor,
        *,
        names: Sequence[AuxiliaryStateTargetName] | None = None,
    ) -> dict[str, object]:
        predictions = self._auxiliary_state_heads.predict_values(latent, names=names)
        return {str(name): value for name, value in predictions.items()}


def _recurrent_tensor_state(
    *,
    policy: MaskableRecurrentMultiInputActorCriticPolicy
    | MaskableHybridRecurrentMultiInputActorCriticPolicy,
    state: tuple[np.ndarray, ...] | None,
    obs: PyTorchObs,
    episode_start: np.ndarray | None,
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
            activation_fn=self.activation_fn,
        )

    def evaluate_actions_with_aux(
        self,
        obs: PyTorchObs,
        actions: torch.Tensor,
        *,
        action_masks: MaybeMasks = None,
        auxiliary_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, PolicyAuxiliaryLoss | None]:
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
        aux_loss = self._auxiliary_state_loss(
            source_latent,
            obs=obs,
            sample_mask=auxiliary_mask,
        )
        return values, log_prob, entropy, aux_loss

    def predict_auxiliary_state(
        self,
        observation: np.ndarray | dict[str, np.ndarray],
        *,
        state: tuple[np.ndarray, ...] | None = None,
        episode_start: np.ndarray | None = None,
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
        observation: np.ndarray | dict[str, np.ndarray],
        *,
        state: tuple[np.ndarray, ...] | None = None,
        episode_start: np.ndarray | None = None,
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
        observation: np.ndarray | dict[str, np.ndarray],
        *,
        state: tuple[np.ndarray, ...] | None = None,
        episode_start: np.ndarray | None = None,
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
