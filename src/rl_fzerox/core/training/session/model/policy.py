# src/rl_fzerox/core/training/session/model/policy.py
from __future__ import annotations

from rl_fzerox.core.config.schema import PolicyConfig


def resolve_policy_name(*, train_env, recurrent_enabled: bool) -> str:
    """Select the SB3 policy class name for the env observation shape."""

    from gymnasium import spaces

    if isinstance(train_env.observation_space, spaces.Dict):
        return "MultiInputLstmPolicy" if recurrent_enabled else "MultiInputPolicy"
    return "CnnLstmPolicy" if recurrent_enabled else "CnnPolicy"


def build_policy_kwargs(
    *,
    train_env,
    policy_config: PolicyConfig,
    value_head_key: str,
) -> dict[str, object]:
    from gymnasium import spaces

    from rl_fzerox.core.policy import FZeroXImageStateExtractor, FZeroXObservationCnnExtractor

    if isinstance(train_env.observation_space, spaces.Dict):
        extractor_class = FZeroXImageStateExtractor
        extractor_kwargs = {
            "conv_profile": policy_config.extractor.conv_profile,
            "features_dim": policy_config.extractor.features_dim,
            "state_features_dim": policy_config.extractor.state_features_dim,
            "fusion_features_dim": policy_config.extractor.fusion_features_dim,
            "layer_norm": policy_config.extractor.layer_norm,
        }
    else:
        extractor_class = FZeroXObservationCnnExtractor
        extractor_kwargs = {
            "conv_profile": policy_config.extractor.conv_profile,
            "features_dim": policy_config.extractor.features_dim,
            "layer_norm": policy_config.extractor.layer_norm,
        }

    return {
        "features_extractor_class": extractor_class,
        "features_extractor_kwargs": extractor_kwargs,
        "net_arch": {
            "pi": [int(value) for value in policy_config.net_arch.pi],
            value_head_key: [int(value) for value in policy_config.net_arch.vf],
        },
        "activation_fn": resolve_policy_activation_fn(policy_config.activation),
    }


def resolve_policy_activation_fn(name: str):
    """Map the configured SB3 policy-head activation name to a torch module."""

    from torch import nn

    if name == "tanh":
        return nn.Tanh
    if name == "relu":
        return nn.ReLU
    raise ValueError(f"Unsupported policy activation: {name!r}")
