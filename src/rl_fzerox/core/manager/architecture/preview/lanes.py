# src/rl_fzerox/core/manager/architecture/preview/lanes.py
from __future__ import annotations

from rl_fzerox.core.manager.architecture.models import (
    ActionBranchPreview,
    ArchitectureLanePreview,
    ArchitectureNodePreview,
    ShapePreview,
)
from rl_fzerox.core.manager.run_spec import ManagedRunConfig


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
    *,
    action_net_detail: str,
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
                    detail=action_net_detail,
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
