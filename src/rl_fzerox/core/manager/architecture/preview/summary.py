# src/rl_fzerox/core/manager/architecture/preview/summary.py
"""Top-level policy architecture preview assembly."""

from __future__ import annotations

from dataclasses import dataclass

from rl_fzerox.core.manager.architecture.models import (
    ActionBranchPreview,
    ConvLayerPreview,
    ParameterGroupPreview,
    PolicyArchitecturePreview,
    ShapePreview,
    StateFeaturePreview,
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
from rl_fzerox.core.policy.auxiliary_state.targets import auxiliary_state_target_spec


@dataclass(frozen=True, slots=True)
class _PreviewDimensions:
    image_shape: ShapePreview
    state_features: tuple[StateFeaturePreview, ...]
    state_dim: int
    conv_layers: tuple[ConvLayerPreview, ...]
    flatten_dim: int
    image_features_dim: int
    state_features_dim: int
    fusion_input_dim: int
    extractor_output_dim: int
    policy_input_dim: int
    pi_output_dim: int
    vf_output_dim: int
    auxiliary_output_dim: int


@dataclass(frozen=True, slots=True)
class _ActionPreview:
    branches: tuple[ActionBranchPreview, ...]
    continuous_dims: int
    discrete_logits: int
    head_params: int


@dataclass(frozen=True, slots=True)
class _PreviewParams:
    cnn: int
    image_projection: int
    state_mlp: int
    fusion: int
    layer_norm: int
    recurrent: int
    auxiliary_head: int
    pi_head: int
    action_head: int
    vf_head: int
    value_output: int

    @property
    def parameter_groups(self) -> tuple[ParameterGroupPreview, ...]:
        return (
            ParameterGroupPreview(name="CNN convolutions", params=self.cnn),
            ParameterGroupPreview(name="Image projection", params=self.image_projection),
            ParameterGroupPreview(name="State MLP", params=self.state_mlp),
            ParameterGroupPreview(name="Fusion", params=self.fusion),
            ParameterGroupPreview(name="LayerNorm", params=self.layer_norm),
            ParameterGroupPreview(name="LSTM", params=self.recurrent),
            ParameterGroupPreview(name="Aux head", params=self.auxiliary_head),
            ParameterGroupPreview(name="Policy head", params=self.pi_head + self.action_head),
            ParameterGroupPreview(name="Value head", params=self.vf_head + self.value_output),
        )

    @property
    def total(self) -> int:
        return sum(group.params for group in self.parameter_groups)

    def node_params(self) -> dict[str, int]:
        return architecture_node_params(
            cnn_params=self.cnn,
            image_projection_params=self.image_projection,
            state_mlp_params=self.state_mlp,
            fusion_params=self.fusion,
            layer_norm_params=self.layer_norm,
            recurrent_params=self.recurrent,
            pi_head_params=self.pi_head,
            action_head_params=self.action_head,
            auxiliary_head_params=self.auxiliary_head,
            vf_head_params=self.vf_head,
            value_output_params=self.value_output,
        )


def policy_architecture_preview(config: ManagedRunConfig) -> PolicyArchitecturePreview:
    """Estimate policy shape and trainable parameters from one managed config."""

    dimensions = _preview_dimensions(config)
    action = _action_preview(config, dimensions)
    params = _preview_params(config, dimensions, action)
    parameter_groups = params.parameter_groups
    return PolicyArchitecturePreview(
        image_shape=dimensions.image_shape,
        state_dim=dimensions.state_dim,
        state_features=dimensions.state_features,
        conv_layers=dimensions.conv_layers,
        flatten_dim=dimensions.flatten_dim,
        image_features_dim=dimensions.image_features_dim,
        state_features_dim=dimensions.state_features_dim,
        fusion_input_dim=dimensions.fusion_input_dim,
        extractor_output_dim=dimensions.extractor_output_dim,
        policy_input_dim=dimensions.policy_input_dim,
        action_branches=action.branches,
        continuous_action_dims=action.continuous_dims,
        discrete_action_logits=action.discrete_logits,
        parameter_groups=parameter_groups,
        total_params=params.total,
        architecture_lanes=architecture_lanes(
            config,
            dimensions.image_shape,
            dimensions.state_dim,
            dimensions.flatten_dim,
            dimensions.fusion_input_dim,
            dimensions.extractor_output_dim,
            dimensions.policy_input_dim,
            dimensions.pi_output_dim,
            dimensions.vf_output_dim,
            dimensions.auxiliary_output_dim,
            action.branches,
            action.continuous_dims,
            action.discrete_logits,
            params.node_params(),
            action_net_detail=action_net_detail(
                pi_output_dim=dimensions.pi_output_dim,
                action_branches=action.branches,
                continuous_action_dims=action.continuous_dims,
                discrete_action_logits=action.discrete_logits,
            ),
        ),
    )


def _preview_dimensions(config: ManagedRunConfig) -> _PreviewDimensions:
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
    state_net_arch = tuple(int(width) for width in config.policy.state_net_arch)
    state_features_dim = state_net_arch[-1] if state_net_arch else state_dim
    fusion_input_dim = image_features_dim + state_features_dim
    extractor_output_dim = (
        fusion_input_dim
        if config.policy.fusion_features_dim is None
        else int(config.policy.fusion_features_dim)
    )
    policy_input_dim = (
        int(config.policy.recurrent_hidden_size)
        if config.policy.recurrent_enabled
        else extractor_output_dim
    )
    pi_output_dim = config.policy.pi_net_arch[-1] if config.policy.pi_net_arch else policy_input_dim
    vf_output_dim = config.policy.vf_net_arch[-1] if config.policy.vf_net_arch else policy_input_dim
    return _PreviewDimensions(
        image_shape=image_shape,
        state_features=state_features,
        state_dim=state_dim,
        conv_layers=conv_layers,
        flatten_dim=flatten_dim,
        image_features_dim=image_features_dim,
        state_features_dim=state_features_dim,
        fusion_input_dim=fusion_input_dim,
        extractor_output_dim=extractor_output_dim,
        policy_input_dim=policy_input_dim,
        pi_output_dim=int(pi_output_dim),
        vf_output_dim=int(vf_output_dim),
        auxiliary_output_dim=auxiliary_state_target_spec().count,
    )


def _preview_params(
    config: ManagedRunConfig,
    dimensions: _PreviewDimensions,
    action: _ActionPreview,
) -> _PreviewParams:
    auxiliary_head_arch = tuple(int(width) for width in config.policy.auxiliary_state_head_arch)
    auxiliary_head_output_dim = (
        auxiliary_head_arch[-1] if auxiliary_head_arch else dimensions.policy_input_dim
    )
    auxiliary_head_params = (
        mlp_params(dimensions.policy_input_dim, auxiliary_head_arch)
        + linear_params(auxiliary_head_output_dim, dimensions.auxiliary_output_dim)
        if config.policy.auxiliary_state_enabled
        else 0
    )
    return _PreviewParams(
        cnn=sum(layer.params for layer in dimensions.conv_layers),
        image_projection=(
            0
            if config.policy.features_dim == "auto"
            else linear_params(dimensions.flatten_dim, dimensions.image_features_dim)
        ),
        state_mlp=mlp_params(
            dimensions.state_dim,
            tuple(int(width) for width in config.policy.state_net_arch),
        ),
        fusion=(
            0
            if config.policy.fusion_features_dim is None
            else linear_params(dimensions.fusion_input_dim, dimensions.extractor_output_dim)
        ),
        layer_norm=dimensions.extractor_output_dim * 2 if config.policy.layer_norm else 0,
        recurrent=recurrent_param_count(config, dimensions.extractor_output_dim),
        auxiliary_head=auxiliary_head_params,
        pi_head=mlp_params(dimensions.policy_input_dim, config.policy.pi_net_arch),
        action_head=action.head_params,
        vf_head=mlp_params(dimensions.policy_input_dim, config.policy.vf_net_arch),
        value_output=linear_params(dimensions.vf_output_dim, 1),
    )


def _action_preview(config: ManagedRunConfig, dimensions: _PreviewDimensions) -> _ActionPreview:
    action_branches = action_branch_previews(config)
    continuous_action_dims = sum(
        branch.size for branch in action_branches if branch.kind == "continuous"
    )
    discrete_action_logits = sum(
        branch.size for branch in action_branches if branch.kind == "discrete"
    )
    action_head_params = linear_params(
        dimensions.pi_output_dim,
        continuous_action_dims,
    ) + linear_params(
        dimensions.pi_output_dim,
        discrete_action_logits,
    )
    return _ActionPreview(
        branches=action_branches,
        continuous_dims=continuous_action_dims,
        discrete_logits=discrete_action_logits,
        head_params=action_head_params,
    )
