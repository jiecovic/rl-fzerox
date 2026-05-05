# src/rl_fzerox/core/manager/architecture/preview.py
from __future__ import annotations

from rl_fzerox.core.manager.architecture.metadata import component_features, preset_geometry
from rl_fzerox.core.manager.architecture.models import (
    ActionBranchPreview,
    ArchitectureLanePreview,
    ArchitectureNodePreview,
    ConvLayerPreview,
    ParameterGroupPreview,
    PolicyArchitecturePreview,
    ShapePreview,
    StateFeaturePreview,
)
from rl_fzerox.core.manager.config import ConvProfile, ManagedRunConfig
from rl_fzerox.core.policy.extractors import conv_output_size, resolve_conv_spec


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
    extractor_output_dim = int(config.policy.fusion_features_dim)
    fusion_params = linear_params(fusion_input_dim, extractor_output_dim)
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
        ParameterGroupPreview(
            name="CNN convolutions",
            params=cnn_params,
        ),
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
        ),
    )


def image_shape_preview(config: ManagedRunConfig) -> ShapePreview:
    height, width = preset_geometry(config.observation.preset)
    channels_per_frame = STACK_MODE_CHANNELS[config.observation.stack_mode]
    channels = (channels_per_frame * int(config.observation.frame_stack)) + (
        1 if config.observation.minimap_layer else 0
    )
    return ShapePreview(height=height, width=width, channels=channels)


def state_feature_previews(config: ManagedRunConfig) -> tuple[StateFeaturePreview, ...]:
    feature_dropouts = {
        feature.name: float(feature.dropout_prob)
        for feature in config.observation.state_feature_dropouts
    }
    independent_lean_buttons = config.action.lean_output_mode == "independent_buttons"
    features: list[StateFeaturePreview] = []
    for component in config.observation.state_components:
        for feature in component_features(
            component,
            independent_lean_buttons=independent_lean_buttons,
        ):
            features.append(
                StateFeaturePreview(
                    component=component.name,
                    name=feature.name,
                    dropout_prob=feature_dropouts.get(feature.name, 0.0),
                )
            )
    return tuple(features)


def conv_layer_previews(
    *,
    height: int,
    width: int,
    channels: int,
    conv_profile: ConvProfile,
    custom_conv_layers: tuple[dict[str, int], ...] | None = None,
) -> tuple[tuple[ConvLayerPreview, ...], int]:
    layers: list[ConvLayerPreview] = []
    in_channels = channels
    output_height = height
    output_width = width
    for index, layer in enumerate(
        resolve_conv_spec(
            (height, width),
            conv_profile=conv_profile,
            custom_conv_layers=custom_conv_layers,
        ),
        start=1,
    ):
        kernel_size = int(layer.kernel_size[0])
        stride = int(layer.stride[0])
        padding = int(layer.padding[0])
        input_height = output_height
        input_width = output_width
        output_height = conv_output_size(input_height, kernel_size, stride, padding)
        output_width = conv_output_size(input_width, kernel_size, stride, padding)
        dropped_height = dropped_trailing_pixels(
            input_size=input_height + (2 * padding),
            kernel_size=kernel_size,
            stride=stride,
            output_size=output_height,
        )
        dropped_width = dropped_trailing_pixels(
            input_size=input_width + (2 * padding),
            kernel_size=kernel_size,
            stride=stride,
            output_size=output_width,
        )
        params = conv_params(
            in_channels=in_channels,
            out_channels=layer.out_channels,
            kernel_size=kernel_size,
        )
        layers.append(
            ConvLayerPreview(
                name=f"conv{index}",
                in_channels=in_channels,
                out_channels=layer.out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                input_height=input_height,
                input_width=input_width,
                output_height=output_height,
                output_width=output_width,
                dropped_height=dropped_height,
                dropped_width=dropped_width,
                params=params,
            )
        )
        in_channels = layer.out_channels
    return tuple(layers), in_channels * output_height * output_width


def recurrent_param_count(config: ManagedRunConfig, input_dim: int) -> int:
    if not config.policy.recurrent_enabled:
        return 0
    actor_params = lstm_params(
        input_dim=input_dim,
        hidden_size=int(config.policy.recurrent_hidden_size),
        layers=int(config.policy.recurrent_n_lstm_layers),
    )
    if config.policy.recurrent_shared_lstm or not config.policy.recurrent_enable_critic_lstm:
        return actor_params
    return actor_params * 2


def architecture_lanes(
    config: ManagedRunConfig,
    image_shape: ShapePreview,
    state_dim: int,
    flatten_dim: int,
    fusion_input_dim: int,
    extractor_output_dim: int,
    policy_input_dim: int,
    pi_output_dim: int,
    vf_output_dim: int,
    action_branches: tuple[ActionBranchPreview, ...],
    continuous_action_dims: int,
    discrete_action_logits: int,
    node_params: dict[str, int],
) -> tuple[ArchitectureLanePreview, ...]:
    image_projection_detail = (
        f"identity {flatten_dim}"
        if config.policy.features_dim == "auto"
        else f"{flatten_dim} → {config.policy.features_dim}"
    )
    image_projection_tone = "muted" if config.policy.features_dim == "auto" else "normal"
    layer_norm_detail = "on" if config.policy.layer_norm else "off"
    layer_norm_tone = "normal" if config.policy.layer_norm else "muted"
    recurrent_detail = (
        recurrent_detail_text(config, extractor_output_dim)
        if config.policy.recurrent_enabled
        else "off"
    )
    recurrent_tone = "normal" if config.policy.recurrent_enabled else "muted"
    return (
        ArchitectureLanePreview(
            id="image_branch",
            label="Image branch",
            nodes=(
                ArchitectureNodePreview(
                    id="image",
                    label="Image",
                    detail=f"{image_shape.height} x {image_shape.width} x {image_shape.channels}",
                ),
                ArchitectureNodePreview(
                    id="cnn",
                    label="CNN",
                    detail=f"{config.policy.conv_profile} → {flatten_dim}",
                    params=node_params.get("cnn"),
                ),
                ArchitectureNodePreview(
                    id="image_projection",
                    label="Image projection",
                    detail=image_projection_detail,
                    params=node_params.get("image_projection"),
                    tone=image_projection_tone,
                ),
            ),
        ),
        ArchitectureLanePreview(
            id="state_branch",
            label="State branch",
            nodes=(
                ArchitectureNodePreview(
                    id="state",
                    label="State vector",
                    detail=f"{state_dim} scalars",
                ),
                ArchitectureNodePreview(
                    id="state_mlp",
                    label="State MLP",
                    detail=state_mlp_detail(config, state_dim),
                    params=node_params.get("state_mlp"),
                    tone="normal" if config.policy.state_net_arch else "muted",
                ),
            ),
        ),
        ArchitectureLanePreview(
            id="fusion_and_heads",
            label="Fusion and heads",
            nodes=(
                ArchitectureNodePreview(
                    id="concat",
                    label="Concat",
                    detail=f"{fusion_input_dim}",
                ),
                ArchitectureNodePreview(
                    id="fusion",
                    label="Fusion MLP",
                    detail=f"{fusion_input_dim} → {extractor_output_dim}",
                    params=node_params.get("fusion"),
                ),
                ArchitectureNodePreview(
                    id="layer_norm",
                    label="LayerNorm",
                    detail=layer_norm_detail,
                    params=node_params.get("layer_norm"),
                    tone=layer_norm_tone,
                ),
                ArchitectureNodePreview(
                    id="lstm",
                    label="LSTM",
                    detail=recurrent_detail,
                    params=node_params.get("lstm"),
                    tone=recurrent_tone,
                ),
                ArchitectureNodePreview(
                    id="policy_head",
                    label="Policy head",
                    detail=(
                        f"{policy_input_dim} → {list(config.policy.pi_net_arch)}, "
                        f"{config.policy.activation}"
                    ),
                    params=node_params.get("policy_head"),
                ),
                ArchitectureNodePreview(
                    id="action_net",
                    label="Action net",
                    detail=action_net_detail(
                        pi_output_dim=pi_output_dim,
                        action_branches=action_branches,
                        continuous_action_dims=continuous_action_dims,
                        discrete_action_logits=discrete_action_logits,
                    ),
                    params=node_params.get("action_net"),
                ),
                ArchitectureNodePreview(
                    id="value_head",
                    label="Value head",
                    detail=(
                        f"{policy_input_dim} → {list(config.policy.vf_net_arch)}, "
                        f"{config.policy.activation}"
                    ),
                    params=node_params.get("value_head"),
                ),
                ArchitectureNodePreview(
                    id="value_net",
                    label="Value net",
                    detail=f"{vf_output_dim} → 1 value",
                    params=node_params.get("value_net"),
                ),
            ),
        ),
    )


def architecture_node_params(
    *,
    cnn_params: int,
    image_projection_params: int,
    state_mlp_params: int,
    fusion_params: int,
    layer_norm_params: int,
    recurrent_params: int,
    pi_head_params: int,
    action_head_params: int,
    vf_head_params: int,
    value_output_params: int,
) -> dict[str, int]:
    return {
        "cnn": cnn_params,
        "image_projection": image_projection_params,
        "state_mlp": state_mlp_params,
        "fusion": fusion_params,
        "layer_norm": layer_norm_params,
        "lstm": recurrent_params,
        "policy_head": pi_head_params,
        "action_net": action_head_params,
        "value_head": vf_head_params,
        "value_net": value_output_params,
    }


def state_mlp_detail(config: ManagedRunConfig, state_dim: int) -> str:
    if not config.policy.state_net_arch:
        return f"identity {state_dim}"
    return f"{state_dim} → {list(config.policy.state_net_arch)}"


def recurrent_detail_text(config: ManagedRunConfig, extractor_output_dim: int) -> str:
    if config.policy.recurrent_shared_lstm:
        topology = "shared"
    elif config.policy.recurrent_enable_critic_lstm:
        topology = "actor + critic"
    else:
        topology = "actor only"
    return (
        f"{extractor_output_dim} → {config.policy.recurrent_hidden_size}, "
        f"{config.policy.recurrent_n_lstm_layers} layer, {topology}"
    )


def linear_params(in_features: int, out_features: int) -> int:
    if out_features <= 0:
        return 0
    return (in_features * out_features) + out_features


def action_branch_previews(config: ManagedRunConfig) -> tuple[ActionBranchPreview, ...]:
    branches: list[ActionBranchPreview] = []
    if config.action.steering_mode == "continuous":
        branches.append(ActionBranchPreview(name="steer", kind="continuous", size=1, enabled=True))
    else:
        branches.append(
            ActionBranchPreview(
                name="steer",
                kind="discrete",
                size=int(config.action.steer_buckets),
                enabled=True,
            )
        )
    if config.action.drive_mode == "pwm":
        branches.append(
            ActionBranchPreview(
                name="throttle",
                kind="continuous",
                size=1,
                enabled=not config.action.force_full_throttle,
                mask_label=None if not config.action.force_full_throttle else "forced full",
            )
        )
    else:
        branches.append(
            ActionBranchPreview(
                name="throttle",
                kind="discrete",
                size=2,
                enabled=not config.action.force_full_throttle,
                mask_label=None if not config.action.force_full_throttle else "forced engaged",
            )
        )
    if config.action.include_air_brake:
        air_brake_mask_label = None
        if not config.action.enable_air_brake:
            air_brake_mask_label = "masked idle"
        elif config.action.mask_air_brake_on_ground:
            air_brake_mask_label = "air-only"
        if config.action.air_brake_mode == "pwm":
            branches.append(
                ActionBranchPreview(
                    name="air_brake",
                    kind="continuous",
                    size=1,
                    enabled=config.action.enable_air_brake,
                    mask_label=air_brake_mask_label,
                )
            )
        else:
            branches.append(
                ActionBranchPreview(
                    name="air_brake",
                    kind="discrete",
                    size=2,
                    enabled=config.action.enable_air_brake,
                    mask_label=air_brake_mask_label,
                )
            )
    if config.action.include_boost:
        boost_mask_label = None
        if not config.action.enable_boost:
            boost_mask_label = "masked idle"
        else:
            boost_guards: list[str] = []
            if config.action.boost_unmask_max_speed_kph is not None:
                boost_guards.insert(0, f"≤ {config.action.boost_unmask_max_speed_kph:g} kph")
            if config.action.boost_min_energy_fraction > 0:
                boost_guards.append(f"≥ {config.action.boost_min_energy_fraction * 100:g}% energy")
            boost_mask_label = ", ".join(boost_guards) if boost_guards else None
        branches.append(
            ActionBranchPreview(
                name="boost",
                kind="discrete",
                size=2,
                enabled=config.action.enable_boost,
                mask_label=boost_mask_label,
            )
        )
    if config.action.include_lean:
        lean_mask_label = None
        if not config.action.enable_lean:
            lean_mask_label = "masked idle"
        elif config.action.lean_unmask_min_speed_kph is not None:
            lean_mask_label = f"≥ {config.action.lean_unmask_min_speed_kph:g} kph"
        if config.action.lean_output_mode == "independent_buttons":
            branches.extend(
                (
                    ActionBranchPreview(
                        name="lean_left",
                        kind="discrete",
                        size=2,
                        enabled=config.action.enable_lean,
                        mask_label=lean_mask_label,
                    ),
                    ActionBranchPreview(
                        name="lean_right",
                        kind="discrete",
                        size=2,
                        enabled=config.action.enable_lean,
                        mask_label=lean_mask_label,
                    ),
                )
            )
        else:
            branches.append(
                ActionBranchPreview(
                    name="lean",
                    kind="discrete",
                    size=3,
                    enabled=config.action.enable_lean,
                    mask_label=lean_mask_label,
                )
            )
    if config.action.include_pitch:
        if config.action.pitch_mode == "continuous":
            branches.append(
                ActionBranchPreview(
                    name="pitch",
                    kind="continuous",
                    size=1,
                    enabled=True,
                )
            )
        else:
            branches.append(
                ActionBranchPreview(
                    name="pitch",
                    kind="discrete",
                    size=int(config.action.pitch_buckets),
                    enabled=config.action.enable_pitch,
                    mask_label=None if config.action.enable_pitch else "masked neutral",
                )
            )
    return tuple(branches)


def action_net_detail(
    *,
    pi_output_dim: int,
    action_branches: tuple[ActionBranchPreview, ...],
    continuous_action_dims: int,
    discrete_action_logits: int,
) -> str:
    branch_details = ", ".join(action_branch_detail(branch) for branch in action_branches)
    return (
        f"{pi_output_dim} → {continuous_action_dims} continuous, "
        f"{discrete_action_logits} logits ({branch_details})"
    )


def action_branch_detail(branch: ActionBranchPreview) -> str:
    detail = f"{branch.name} {branch.size}"
    if branch.mask_label is not None:
        return f"{detail} {branch.mask_label}"
    return detail


def conv_params(*, in_channels: int, out_channels: int, kernel_size: int) -> int:
    return (in_channels * out_channels * kernel_size * kernel_size) + out_channels


def dropped_trailing_pixels(
    *,
    input_size: int,
    kernel_size: int,
    stride: int,
    output_size: int,
) -> int:
    covered_extent = ((output_size - 1) * stride) + kernel_size
    return input_size - covered_extent


def mlp_params(input_dim: int, layers: tuple[int, ...]) -> int:
    total = 0
    current = input_dim
    for layer_width in layers:
        width = int(layer_width)
        total += linear_params(current, width)
        current = width
    return total


def lstm_params(*, input_dim: int, hidden_size: int, layers: int) -> int:
    total = 0
    current_input = input_dim
    for _ in range(layers):
        total += 4 * hidden_size * (current_input + hidden_size + 2)
        current_input = hidden_size
    return total


STACK_MODE_CHANNELS = {"rgb": 3, "gray": 1, "luma_chroma": 2}
