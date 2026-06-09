# src/rl_fzerox/core/training/session/model/policy.py
from __future__ import annotations

from stable_baselines3.common.vec_env import VecEnv

from rl_fzerox.core.domain.training_algorithms import TRAINING_ALGORITHMS
from rl_fzerox.core.policy.activations import resolve_policy_activation_fn
from rl_fzerox.core.policy.auxiliary_state.policies import (
    AuxiliaryStateMaskableHybridActionMultiInputPolicy,
    AuxiliaryStateMaskableHybridRecurrentMultiInputPolicy,
    AuxiliaryStateMaskableRecurrentMultiInputPolicy,
)
from rl_fzerox.core.runtime_spec.schema import EnvConfig, PolicyConfig, TrainConfig


def resolve_policy_entry(
    *,
    train_env: VecEnv,
    effective_algorithm: str,
    policy_config: PolicyConfig,
    train_config: TrainConfig,
    recurrent_enabled: bool,
):
    """Select the SB3 policy class name for the env observation shape."""

    from gymnasium import spaces

    actor_regularization_enabled = train_config.actor_regularization.enabled()
    if actor_regularization_enabled and effective_algorithm not in TRAINING_ALGORITHMS.hybrid:
        raise RuntimeError("train.actor_regularization requires a hybrid action algorithm")

    auxiliary_state_enabled = _needs_auxiliary_policy(policy_config, train_config)
    if auxiliary_state_enabled:
        if not isinstance(train_env.observation_space, spaces.Dict):
            raise RuntimeError(
                "policy auxiliary state or actor regularization requires a dict observation space"
            )
        if effective_algorithm == TRAINING_ALGORITHMS.maskable_recurrent_ppo:
            return AuxiliaryStateMaskableRecurrentMultiInputPolicy
        if effective_algorithm == TRAINING_ALGORITHMS.maskable_hybrid_action_ppo:
            return AuxiliaryStateMaskableHybridActionMultiInputPolicy
        if effective_algorithm == TRAINING_ALGORITHMS.maskable_hybrid_recurrent_ppo:
            return AuxiliaryStateMaskableHybridRecurrentMultiInputPolicy
        raise RuntimeError(
            "policy auxiliary state or actor regularization is not supported for "
            f"train.algorithm={effective_algorithm}"
        )

    if isinstance(train_env.observation_space, spaces.Dict):
        return "MultiInputLstmPolicy" if recurrent_enabled else "MultiInputPolicy"
    return "CnnLstmPolicy" if recurrent_enabled else "CnnPolicy"


def build_policy_kwargs(
    *,
    train_env: VecEnv,
    policy_config: PolicyConfig,
    train_config: TrainConfig,
    env_config: EnvConfig,
    value_head_key: str,
) -> dict[str, object]:
    from gymnasium import spaces

    from rl_fzerox.core.policy import FZeroXImageStateExtractor, FZeroXObservationCnnExtractor

    if isinstance(train_env.observation_space, spaces.Dict):
        extractor_class = FZeroXImageStateExtractor
        extractor_kwargs = {
            "conv_profile": policy_config.extractor.conv_profile,
            "custom_conv_layers": tuple(
                layer.model_dump(mode="python")
                for layer in policy_config.extractor.custom_conv_layers
            ),
            "features_dim": policy_config.extractor.features_dim,
            "image_projection_activation": policy_config.extractor.image_projection_activation,
            "state_features_dim": policy_config.extractor.state_features_dim,
            "state_net_arch": policy_config.extractor.resolved_state_net_arch(),
            "state_activation": policy_config.extractor.state_activation,
            "fusion_features_dim": policy_config.extractor.fusion_features_dim,
            "fusion_activation": policy_config.extractor.fusion_activation,
            "layer_norm": policy_config.extractor.layer_norm,
            "layer_norm_activation": policy_config.extractor.layer_norm_activation,
        }
    else:
        extractor_class = FZeroXObservationCnnExtractor
        extractor_kwargs = {
            "conv_profile": policy_config.extractor.conv_profile,
            "custom_conv_layers": tuple(
                layer.model_dump(mode="python")
                for layer in policy_config.extractor.custom_conv_layers
            ),
            "features_dim": policy_config.extractor.features_dim,
            "image_projection_activation": policy_config.extractor.image_projection_activation,
            "layer_norm": policy_config.extractor.layer_norm,
            "layer_norm_activation": policy_config.extractor.layer_norm_activation,
        }

    policy_kwargs: dict[str, object] = {
        "features_extractor_class": extractor_class,
        "features_extractor_kwargs": extractor_kwargs,
        "net_arch": {
            "pi": [int(value) for value in policy_config.net_arch.pi],
            value_head_key: [int(value) for value in policy_config.net_arch.vf],
        },
        "activation_fn": resolve_policy_activation_fn(policy_config.activation),
    }
    if policy_config.auxiliary_state.enabled:
        policy_kwargs["auxiliary_state"] = policy_config.auxiliary_state.model_dump(mode="python")
    if env_config.action.layout_continuous_axes:
        policy_kwargs["hybrid_action_group_names"] = {
            "continuous": tuple(env_config.action.layout_continuous_axes),
            "discrete": tuple(env_config.action.layout_discrete_axes),
        }
    if train_config.actor_regularization.enabled():
        policy_kwargs["actor_regularization"] = train_config.actor_regularization.model_dump(
            mode="python"
        )
        continuous_action_group_names = tuple(env_config.action.layout_continuous_axes)
        policy_kwargs["continuous_action_group_names"] = continuous_action_group_names
        policy_kwargs["discrete_action_group_names"] = tuple(env_config.action.layout_discrete_axes)
        policy_kwargs["pitch_bucket_count"] = int(env_config.action.pitch_buckets)
    return policy_kwargs


def _needs_auxiliary_policy(
    policy_config: PolicyConfig,
    train_config: TrainConfig,
) -> bool:
    return policy_config.auxiliary_state.enabled or train_config.actor_regularization.enabled()
