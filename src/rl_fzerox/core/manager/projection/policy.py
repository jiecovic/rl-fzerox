# src/rl_fzerox/core/manager/projection/policy.py
"""Managed-run to training policy-config projection."""

from __future__ import annotations

from rl_fzerox.core.manager.run_spec import ManagedRunConfig
from rl_fzerox.core.runtime_spec.schema import TrainAppConfig
from rl_fzerox.core.training.session.model.compatibility import resume_policy_signature


def build_policy_data(config: ManagedRunConfig) -> dict[str, object]:
    return {
        "activation": config.policy.activation,
        "extractor": {
            "conv_profile": config.policy.conv_profile,
            "custom_conv_layers": [
                layer.model_dump(mode="python") for layer in config.policy.custom_conv_layers
            ],
            "features_dim": config.policy.features_dim,
            "image_projection_activation": config.policy.image_projection_activation,
            "state_net_arch": list(config.policy.state_net_arch),
            "state_activation": config.policy.state_activation,
            "fusion_features_dim": config.policy.fusion_features_dim,
            "fusion_activation": config.policy.fusion_activation,
            "layer_norm": config.policy.layer_norm,
            "layer_norm_activation": config.policy.layer_norm_activation,
        },
        "recurrent": {
            "enabled": config.policy.recurrent_enabled,
            "hidden_size": config.policy.recurrent_hidden_size,
            "n_lstm_layers": config.policy.recurrent_n_lstm_layers,
            "shared_lstm": config.policy.recurrent_shared_lstm,
            "enable_critic_lstm": config.policy.recurrent_enable_critic_lstm,
        },
        "action_bias": {
            "gas_on_logit": (
                config.policy.gas_on_logit if config.action.drive_mode == "on_off" else 0.0
            ),
            "air_brake_on_logit": (
                config.policy.air_brake_on_logit
                if config.action.include_air_brake and config.action.air_brake_mode == "on_off"
                else 0.0
            ),
            "spin_idle_logit": (
                config.policy.spin_idle_logit if config.action.include_spin else 0.0
            ),
        },
        "auxiliary_state": {
            "enabled": config.policy.auxiliary_state_enabled,
            "head_arch": list(config.policy.auxiliary_state_head_arch),
            "losses": [
                {
                    "name": loss.name,
                    "weight": float(loss.weight),
                    "grounded_only": bool(loss.grounded_only),
                }
                for loss in config.policy.auxiliary_state_losses
            ],
        },
        "net_arch": {
            "pi": list(config.policy.pi_net_arch),
            "vf": list(config.policy.vf_net_arch),
        },
    }


def fork_policy_signature(train_config: TrainAppConfig) -> dict[str, object]:
    return resume_policy_signature(train_config)
