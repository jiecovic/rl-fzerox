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
from rl_fzerox.core.runtime_spec.schema import PolicyConfig


def resolve_policy_entry(
    *,
    train_env: VecEnv,
    effective_algorithm: str,
    policy_config: PolicyConfig,
    recurrent_enabled: bool,
):
    """Select the SB3 policy class name for the env observation shape."""

    from gymnasium import spaces

    auxiliary_state_enabled = policy_config.auxiliary_state.enabled
    if auxiliary_state_enabled:
        if not isinstance(train_env.observation_space, spaces.Dict):
            raise RuntimeError("policy auxiliary state requires a dict observation space")
        if effective_algorithm == TRAINING_ALGORITHMS.maskable_recurrent_ppo:
            return AuxiliaryStateMaskableRecurrentMultiInputPolicy
        if effective_algorithm == TRAINING_ALGORITHMS.maskable_hybrid_action_ppo:
            return AuxiliaryStateMaskableHybridActionMultiInputPolicy
        if effective_algorithm == TRAINING_ALGORITHMS.maskable_hybrid_recurrent_ppo:
            return AuxiliaryStateMaskableHybridRecurrentMultiInputPolicy
        raise RuntimeError(
            f"policy auxiliary state is not supported for train.algorithm={effective_algorithm}"
        )

    if isinstance(train_env.observation_space, spaces.Dict):
        return "MultiInputLstmPolicy" if recurrent_enabled else "MultiInputPolicy"
    return "CnnLstmPolicy" if recurrent_enabled else "CnnPolicy"


def build_policy_kwargs(
    *,
    train_env: VecEnv,
    policy_config: PolicyConfig,
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
    return policy_kwargs
