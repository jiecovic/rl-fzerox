# src/rl_fzerox/core/manager/architecture/preview/summary.py
from __future__ import annotations

from rl_fzerox.core.manager.architecture.models import (
    ParameterGroupPreview,
    PolicyArchitecturePreview,
)
from rl_fzerox.core.manager.architecture.preview.actions import (
    action_branch_previews,
    action_net_detail,
)
from rl_fzerox.core.manager.architecture.preview.lanes import (
    architecture_lanes,
    architecture_node_params,
)
from rl_fzerox.core.manager.architecture.preview.params import (
    conv_layer_previews,
    linear_params,
    mlp_params,
    recurrent_param_count,
)
from rl_fzerox.core.manager.architecture.preview.state import (
    image_shape_preview,
    state_feature_previews,
)
from rl_fzerox.core.manager.run_spec import ManagedRunConfig


def policy_architecture_preview(config: ManagedRunConfig) -> PolicyArchitecturePreview:
    """Estimate policy shape and trainable parameters from one managed config."""

    image_shape = image_shape_preview(config)
    state_features = state_feature_previews(config)
    state_dim = len(state_features)
    conv_layers, flatten_dim = conv_layer_previews(
        height=image_shape.height,
        width=image_shape.width,
        channels=image_shape.channels,
        conv_profile=config.policy.conv_profile,
        custom_conv_layers=tuple(
            layer.model_dump(mode="python") for layer in config.policy.custom_conv_layers
        ),
    )
    image_features_dim = (
        flatten_dim if config.policy.features_dim == "auto" else int(config.policy.features_dim)
    )
    image_projection_params = (
        0
        if config.policy.features_dim == "auto"
        else linear_params(flatten_dim, image_features_dim)
    )
    cnn_params = sum(layer.params for layer in conv_layers)
    state_net_arch = tuple(int(width) for width in config.policy.state_net_arch)
    state_features_dim = state_net_arch[-1] if state_net_arch else state_dim
    state_mlp_params = mlp_params(state_dim, state_net_arch)
    fusion_input_dim = image_features_dim + state_features_dim
    extractor_output_dim = (
        fusion_input_dim
        if config.policy.fusion_features_dim is None
        else int(config.policy.fusion_features_dim)
    )
    fusion_params = (
        0
        if config.policy.fusion_features_dim is None
        else linear_params(fusion_input_dim, extractor_output_dim)
    )
    layer_norm_params = extractor_output_dim * 2 if config.policy.layer_norm else 0
    recurrent_params = recurrent_param_count(config, extractor_output_dim)
    policy_input_dim = (
        int(config.policy.recurrent_hidden_size)
        if config.policy.recurrent_enabled
        else extractor_output_dim
    )
    pi_head_params = mlp_params(policy_input_dim, config.policy.pi_net_arch)
    vf_head_params = mlp_params(policy_input_dim, config.policy.vf_net_arch)
    pi_output_dim = config.policy.pi_net_arch[-1] if config.policy.pi_net_arch else policy_input_dim
    vf_output_dim = config.policy.vf_net_arch[-1] if config.policy.vf_net_arch else policy_input_dim
    action_branches = action_branch_previews(config)
    continuous_action_dims = sum(
        branch.size for branch in action_branches if branch.kind == "continuous"
    )
    discrete_action_logits = sum(
        branch.size for branch in action_branches if branch.kind == "discrete"
    )
    action_head_params = linear_params(int(pi_output_dim), continuous_action_dims) + linear_params(
        int(pi_output_dim), discrete_action_logits
    )
    value_output_params = linear_params(int(vf_output_dim), 1)
    parameter_groups = (
        ParameterGroupPreview(name="CNN convolutions", params=cnn_params),
        ParameterGroupPreview(name="Image projection", params=image_projection_params),
        ParameterGroupPreview(name="State MLP", params=state_mlp_params),
        ParameterGroupPreview(name="Fusion", params=fusion_params),
        ParameterGroupPreview(name="LayerNorm", params=layer_norm_params),
        ParameterGroupPreview(name="LSTM", params=recurrent_params),
        ParameterGroupPreview(name="Policy head", params=pi_head_params + action_head_params),
        ParameterGroupPreview(name="Value head", params=vf_head_params + value_output_params),
    )
    total_params = sum(group.params for group in parameter_groups)
    node_params = architecture_node_params(
        cnn_params=cnn_params,
        image_projection_params=image_projection_params,
        state_mlp_params=state_mlp_params,
        fusion_params=fusion_params,
        layer_norm_params=layer_norm_params,
        recurrent_params=recurrent_params,
        pi_head_params=pi_head_params,
        action_head_params=action_head_params,
        vf_head_params=vf_head_params,
        value_output_params=value_output_params,
    )
    return PolicyArchitecturePreview(
        image_shape=image_shape,
        state_dim=state_dim,
        state_features=state_features,
        conv_layers=conv_layers,
        flatten_dim=flatten_dim,
        image_features_dim=image_features_dim,
        state_features_dim=state_features_dim,
        fusion_input_dim=fusion_input_dim,
        extractor_output_dim=extractor_output_dim,
        policy_input_dim=policy_input_dim,
        action_branches=action_branches,
        continuous_action_dims=continuous_action_dims,
        discrete_action_logits=discrete_action_logits,
        parameter_groups=parameter_groups,
        total_params=total_params,
        architecture_lanes=architecture_lanes(
            config,
            image_shape,
            state_dim,
            flatten_dim,
            fusion_input_dim,
            extractor_output_dim,
            policy_input_dim,
            int(pi_output_dim),
            int(vf_output_dim),
            action_branches,
            continuous_action_dims,
            discrete_action_logits,
            node_params,
            action_net_detail=action_net_detail(
                pi_output_dim=int(pi_output_dim),
                action_branches=action_branches,
                continuous_action_dims=continuous_action_dims,
                discrete_action_logits=discrete_action_logits,
            ),
        ),
    )
