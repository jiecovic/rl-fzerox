# src/rl_fzerox/core/manager/architecture.py
from __future__ import annotations

from collections.abc import Iterable

from pydantic import BaseModel, ConfigDict

from rl_fzerox.core.domain.observation_components import ObservationStateComponentName
from rl_fzerox.core.envs.observations.state.components import state_component_definition
from rl_fzerox.core.envs.observations.state.types import StateFeature
from rl_fzerox.core.manager.config import (
    ConvProfile,
    ManagedRunConfig,
    ManagedStateComponentConfig,
    ObservationPreset,
    StateComponentMode,
    default_state_components,
)
from rl_fzerox.core.policy.extractors import conv_output_size, resolve_conv_spec


class SelectOption(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: str
    label: str


class ObservationPresetInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: ObservationPreset
    label: str
    height: int
    width: int


class StateComponentInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: ObservationStateComponentName
    label: str
    default_mode: StateComponentMode
    features: tuple[StateFeatureInfo, ...]


class StateFeatureInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    low: float
    high: float


class RunManagerConfigMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    observation_presets: tuple[ObservationPresetInfo, ...]
    stack_modes: tuple[SelectOption, ...]
    resize_filters: tuple[SelectOption, ...]
    progress_sources: tuple[SelectOption, ...]
    component_modes: tuple[SelectOption, ...]
    action_history_controls: tuple[SelectOption, ...]
    state_components: tuple[StateComponentInfo, ...]
    conv_profiles: tuple[SelectOption, ...]
    activation_functions: tuple[SelectOption, ...]
    net_arch_presets: tuple[SelectOption, ...]


class ShapePreview(BaseModel):
    model_config = ConfigDict(extra="forbid")

    height: int
    width: int
    channels: int


class StateFeaturePreview(BaseModel):
    model_config = ConfigDict(extra="forbid")

    component: ObservationStateComponentName
    name: str
    mode: StateComponentMode


class ConvLayerPreview(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    in_channels: int
    out_channels: int
    kernel_size: int
    stride: int
    output_height: int
    output_width: int
    params: int


class ParameterGroupPreview(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    params: int


class ArchitectureNodePreview(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    label: str
    detail: str
    tone: str = "normal"


class ArchitectureLanePreview(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    label: str
    nodes: tuple[ArchitectureNodePreview, ...]


class PolicyArchitecturePreview(BaseModel):
    model_config = ConfigDict(extra="forbid")

    image_shape: ShapePreview
    state_dim: int
    state_features: tuple[StateFeaturePreview, ...]
    conv_layers: tuple[ConvLayerPreview, ...]
    flatten_dim: int
    image_features_dim: int
    state_features_dim: int
    fusion_input_dim: int
    extractor_output_dim: int
    policy_input_dim: int
    parameter_groups: tuple[ParameterGroupPreview, ...]
    total_params: int
    architecture_lanes: tuple[ArchitectureLanePreview, ...]


def run_manager_config_metadata() -> RunManagerConfigMetadata:
    """Return stable manager options derived from backend-supported config values."""

    return RunManagerConfigMetadata(
        observation_presets=tuple(
            ObservationPresetInfo(
                value=value,
                label=value.replace("crop_", "").replace("x", " x "),
                height=height,
                width=width,
            )
            for value, height, width in OBSERVATION_PRESET_GEOMETRIES
        ),
        stack_modes=_options(("rgb", "gray", "luma_chroma")),
        resize_filters=_options(("nearest", "bilinear")),
        progress_sources=_options(("lap_progress", "segment_progress", "none")),
        component_modes=_options(("include", "zero", "exclude")),
        action_history_controls=_options(
            ("steer", "thrust", "air_brake", "boost", "lean", "pitch")
        ),
        state_components=tuple(
            StateComponentInfo(
                name=component.name,
                label=component.name.replace("_", " "),
                default_mode=component.mode,
                features=tuple(
                    StateFeatureInfo(name=feature.name, low=feature.low, high=feature.high)
                    for feature in _component_features(component)
                ),
            )
            for component in _default_state_components()
        ),
        conv_profiles=_options(
            (
                "auto",
                "nature",
                "nature_32_64_128",
                "nature_wide",
                "nature_extra_k3",
                "compact_deep",
                "compact_bottleneck",
                "tiny_256",
            )
        ),
        activation_functions=_options(("relu", "gelu", "tanh")),
        net_arch_presets=(
            SelectOption(value="256,128", label="[256, 128]"),
            SelectOption(value="512,256", label="[512, 256]"),
            SelectOption(value="256", label="[256]"),
            SelectOption(value="128", label="[128]"),
        ),
    )


def policy_architecture_preview(config: ManagedRunConfig) -> PolicyArchitecturePreview:
    """Estimate policy shape and trainable parameters from one managed config."""

    image_shape = _image_shape(config)
    state_features = _state_features(config)
    state_dim = len(state_features)
    conv_layers, flatten_dim = _conv_layers(
        height=image_shape.height,
        width=image_shape.width,
        channels=image_shape.channels,
        conv_profile=config.policy.conv_profile,
    )
    image_features_dim = (
        flatten_dim if config.policy.features_dim == "auto" else int(config.policy.features_dim)
    )
    image_projection_params = (
        0
        if config.policy.features_dim == "auto"
        else linear_params(flatten_dim, image_features_dim)
    )
    state_features_dim = int(config.policy.state_features_dim)
    state_mlp_params = linear_params(state_dim, state_features_dim)
    fusion_input_dim = image_features_dim + state_features_dim
    extractor_output_dim = int(config.policy.fusion_features_dim)
    fusion_params = linear_params(fusion_input_dim, extractor_output_dim)
    layer_norm_params = extractor_output_dim * 2 if config.policy.layer_norm else 0
    recurrent_params = _recurrent_params(config, extractor_output_dim)
    policy_input_dim = (
        int(config.policy.recurrent_hidden_size)
        if config.policy.recurrent_enabled
        else extractor_output_dim
    )
    pi_head_params = mlp_params(policy_input_dim, config.policy.pi_net_arch)
    vf_head_params = mlp_params(policy_input_dim, config.policy.vf_net_arch)
    pi_output_dim = config.policy.pi_net_arch[-1] if config.policy.pi_net_arch else policy_input_dim
    vf_output_dim = config.policy.vf_net_arch[-1] if config.policy.vf_net_arch else policy_input_dim
    action_head_params = linear_params(int(pi_output_dim), ACTION_CONTINUOUS_DIM) + linear_params(
        int(pi_output_dim), ACTION_DISCRETE_LOGITS
    )
    value_head_params = linear_params(int(vf_output_dim), 1)
    parameter_groups = (
        ParameterGroupPreview(
            name="CNN convolutions",
            params=sum(layer.params for layer in conv_layers),
        ),
        ParameterGroupPreview(name="Image projection", params=image_projection_params),
        ParameterGroupPreview(name="State MLP", params=state_mlp_params),
        ParameterGroupPreview(name="Fusion", params=fusion_params),
        ParameterGroupPreview(name="LayerNorm", params=layer_norm_params),
        ParameterGroupPreview(name="LSTM", params=recurrent_params),
        ParameterGroupPreview(name="Policy head", params=pi_head_params + action_head_params),
        ParameterGroupPreview(name="Value head", params=vf_head_params + value_head_params),
    )
    total_params = sum(group.params for group in parameter_groups)
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
        parameter_groups=parameter_groups,
        total_params=total_params,
        architecture_lanes=_architecture_lanes(
            config,
            image_shape,
            state_dim,
            flatten_dim,
            fusion_input_dim,
            extractor_output_dim,
            policy_input_dim,
        ),
    )


def _image_shape(config: ManagedRunConfig) -> ShapePreview:
    height, width = _preset_geometry(config.observation.preset)
    channels_per_frame = STACK_MODE_CHANNELS[config.observation.stack_mode]
    channels = (channels_per_frame * int(config.observation.frame_stack)) + (
        1 if config.observation.minimap_layer else 0
    )
    return ShapePreview(height=height, width=width, channels=channels)


def _state_features(config: ManagedRunConfig) -> tuple[StateFeaturePreview, ...]:
    feature_modes: dict[str, StateComponentMode] = {
        feature.name: feature.mode for feature in config.observation.state_feature_modes
    }
    features: list[StateFeaturePreview] = []
    for component in config.observation.state_components:
        baseline_mode = component.mode
        for feature in _component_features(component):
            feature_mode = feature_modes.get(feature.name, baseline_mode)
            if feature_mode == "exclude":
                continue
            features.append(
                StateFeaturePreview(
                    component=component.name,
                    name=feature.name,
                    mode=feature_mode,
                )
            )
    return tuple(features)


def _conv_layers(
    *,
    height: int,
    width: int,
    channels: int,
    conv_profile: ConvProfile,
) -> tuple[tuple[ConvLayerPreview, ...], int]:
    layers: list[ConvLayerPreview] = []
    in_channels = channels
    output_height = height
    output_width = width
    for index, layer in enumerate(
        resolve_conv_spec((height, width), conv_profile=conv_profile),
        start=1,
    ):
        kernel_size = int(layer.kernel_size[0])
        stride = int(layer.stride[0])
        output_height = conv_output_size(output_height, kernel_size, stride)
        output_width = conv_output_size(output_width, kernel_size, stride)
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
                output_height=output_height,
                output_width=output_width,
                params=params,
            )
        )
        in_channels = layer.out_channels
    return tuple(layers), in_channels * output_height * output_width


def _recurrent_params(config: ManagedRunConfig, input_dim: int) -> int:
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


def _architecture_lanes(
    config: ManagedRunConfig,
    image_shape: ShapePreview,
    state_dim: int,
    flatten_dim: int,
    fusion_input_dim: int,
    extractor_output_dim: int,
    policy_input_dim: int,
) -> tuple[ArchitectureLanePreview, ...]:
    image_projection_detail = (
        f"identity {flatten_dim}"
        if config.policy.features_dim == "auto"
        else f"{flatten_dim} -> {config.policy.features_dim}"
    )
    image_projection_tone = "muted" if config.policy.features_dim == "auto" else "normal"
    layer_norm_detail = "on" if config.policy.layer_norm else "off"
    layer_norm_tone = "normal" if config.policy.layer_norm else "muted"
    recurrent_detail = (
        _recurrent_detail(config, extractor_output_dim)
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
                    detail=f"{config.policy.conv_profile} -> {flatten_dim}",
                ),
                ArchitectureNodePreview(
                    id="image_projection",
                    label="Image projection",
                    detail=image_projection_detail,
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
                    detail=f"{state_dim} -> {config.policy.state_features_dim}",
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
                    detail=f"{fusion_input_dim} -> {extractor_output_dim}",
                ),
                ArchitectureNodePreview(
                    id="layer_norm",
                    label="LayerNorm",
                    detail=layer_norm_detail,
                    tone=layer_norm_tone,
                ),
                ArchitectureNodePreview(
                    id="lstm",
                    label="LSTM",
                    detail=recurrent_detail,
                    tone=recurrent_tone,
                ),
                ArchitectureNodePreview(
                    id="heads",
                    label="Policy / value heads",
                    detail=(
                        f"{policy_input_dim} -> pi {list(config.policy.pi_net_arch)} / "
                        f"vf {list(config.policy.vf_net_arch)}, {config.policy.activation}"
                    ),
                ),
            ),
        ),
    )


def _recurrent_detail(config: ManagedRunConfig, extractor_output_dim: int) -> str:
    if config.policy.recurrent_shared_lstm:
        topology = "shared"
    elif config.policy.recurrent_enable_critic_lstm:
        topology = "actor + critic"
    else:
        topology = "actor only"
    return (
        f"{extractor_output_dim} -> {config.policy.recurrent_hidden_size}, "
        f"{config.policy.recurrent_n_lstm_layers} layer, {topology}"
    )


def _component_features(component: ManagedStateComponentConfig) -> tuple[StateFeature, ...]:
    settings = component.data()
    return state_component_definition(settings).features(settings)


def _default_state_components() -> tuple[ManagedStateComponentConfig, ...]:
    return default_state_components()


def _options(values: Iterable[str]) -> tuple[SelectOption, ...]:
    return tuple(SelectOption(value=value, label=value.replace("_", " ")) for value in values)


def _preset_geometry(preset: ObservationPreset) -> tuple[int, int]:
    for value, height, width in OBSERVATION_PRESET_GEOMETRIES:
        if value == preset:
            return height, width
    raise ValueError(f"Unsupported observation preset: {preset!r}")


def linear_params(in_features: int, out_features: int) -> int:
    return (in_features * out_features) + out_features


def conv_params(*, in_channels: int, out_channels: int, kernel_size: int) -> int:
    return (in_channels * out_channels * kernel_size * kernel_size) + out_channels


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


OBSERVATION_PRESET_GEOMETRIES: tuple[tuple[ObservationPreset, int, int], ...] = (
    ("crop_84x116", 84, 116),
    ("crop_92x124", 92, 124),
    ("crop_116x164", 116, 164),
    ("crop_98x130", 98, 130),
    ("crop_66x82", 66, 82),
    ("crop_60x76", 60, 76),
    ("crop_68x68", 68, 68),
    ("crop_84x84", 84, 84),
    ("crop_76x100", 76, 100),
    ("crop_64x64", 64, 64),
)
STACK_MODE_CHANNELS = {"rgb": 3, "gray": 1, "luma_chroma": 2}
ACTION_CONTINUOUS_DIM = 1
ACTION_DISCRETE_LOGITS = 14
