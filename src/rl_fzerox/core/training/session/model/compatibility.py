# src/rl_fzerox/core/training/session/model/compatibility.py
"""Structural checkpoint compatibility signatures for training resumes."""

from __future__ import annotations

from rl_fzerox.core.runtime_spec.schema import TrainAppConfig

RESUME_COMPATIBILITY_LABELS = (
    ("algorithm", "training algorithm"),
    ("observation", "observation structure"),
    ("action", "action layout"),
    ("policy", "policy architecture"),
)


def resume_compatibility_signature(train_config: TrainAppConfig) -> dict[str, object]:
    """Return the checkpoint-relevant structure of one resolved training config."""

    runtime_action = train_config.env.action.runtime()
    return {
        "algorithm": train_config.train.algorithm,
        "observation": resume_observation_signature(train_config),
        "action": {
            "name": runtime_action.name,
            "steer_buckets": runtime_action.steer_buckets,
            "pitch_buckets": runtime_action.pitch_buckets,
            "lean_output_mode": runtime_action.lean_output_mode,
            "layout_continuous_axes": tuple(runtime_action.layout_continuous_axes),
            "layout_discrete_axes": tuple(runtime_action.layout_discrete_axes),
        },
        "policy": resume_policy_signature(train_config),
    }


def resume_compatibility_change_labels(
    source_config: TrainAppConfig,
    candidate_config: TrainAppConfig,
) -> tuple[str, ...]:
    """Return human labels for checkpoint-structure changes between configs."""

    source_signature = resume_compatibility_signature(source_config)
    candidate_signature = resume_compatibility_signature(candidate_config)
    return tuple(
        label
        for key, label in RESUME_COMPATIBILITY_LABELS
        if source_signature[key] != candidate_signature[key]
    )


def resume_observation_signature(train_config: TrainAppConfig) -> dict[str, object]:
    observation = train_config.env.observation
    return {
        "mode": observation.mode,
        "resolution": observation.resolution.model_dump(mode="python"),
        "frame_stack": observation.frame_stack,
        "stack_mode": observation.stack_mode,
        "minimap_layer": observation.minimap_layer,
        "state_components": tuple(
            component.model_dump(mode="python") for component in observation.state_components or ()
        ),
    }


def resume_policy_signature(train_config: TrainAppConfig) -> dict[str, object]:
    extractor = train_config.policy.extractor
    recurrent = train_config.policy.recurrent
    net_arch = train_config.policy.net_arch
    return {
        "extractor": {
            "conv_profile": extractor.conv_profile,
            "custom_conv_layers": tuple(
                layer.model_dump(mode="python") for layer in extractor.custom_conv_layers
            ),
            "features_dim": extractor.features_dim,
            "state_net_arch": tuple(extractor.state_net_arch or ()),
            "fusion_features_dim": extractor.fusion_features_dim,
            "layer_norm": extractor.layer_norm,
        },
        "recurrent": {
            "enabled": recurrent.enabled,
            "hidden_size": recurrent.hidden_size,
            "n_lstm_layers": recurrent.n_lstm_layers,
            "shared_lstm": recurrent.shared_lstm,
            "enable_critic_lstm": recurrent.enable_critic_lstm,
        },
        "net_arch": {
            "pi": tuple(net_arch.pi),
            "vf": tuple(net_arch.vf),
        },
    }
