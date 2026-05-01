# src/rl_fzerox/core/manager/architecture_models.py
from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from rl_fzerox.core.domain.observation_components import ObservationStateComponentName
from rl_fzerox.core.manager.config import ObservationPreset, StateComponentMode


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


class StateFeatureInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    low: float
    high: float


class StateComponentInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: ObservationStateComponentName
    label: str
    default_mode: StateComponentMode
    features: tuple[StateFeatureInfo, ...]


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
