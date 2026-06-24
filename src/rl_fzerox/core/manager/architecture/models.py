# src/rl_fzerox/core/manager/architecture/models.py
from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from rl_fzerox.core.domain.observations import ObservationPresetName, ObservationStateComponentName
from rl_fzerox.core.domain.policy import CnnActivationName, CnnLayerKind


class SelectOption(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: str
    label: str


class ObservationPresetInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: ObservationPresetName
    label: str
    height: int
    width: int


class ObservationResolutionBounds(BaseModel):
    model_config = ConfigDict(extra="forbid")

    min_dimension: int
    max_height: int
    max_width: int


class ObservationSourceGeometryInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    renderer: str
    height: int
    width: int


class StateFeatureInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    low: float
    high: float
    default_enabled: bool = True
    auxiliary_target_name: str | None = None
    auxiliary_supports_grounded_only: bool = False


class StateComponentInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: ObservationStateComponentName
    label: str
    features: tuple[StateFeatureInfo, ...]


class TrackCupInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    label: str
    order: int
    course_ids: tuple[str, ...]


class BuiltInCourseInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    ref: str
    display_name: str
    cup: str
    cup_label: str
    course_index: int
    default_selected: bool = True


class VehicleInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    display_name: str
    character_index: int
    machine_select_slot: int
    menu_row: int
    menu_column: int


class RunManagerConfigMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    observation_presets: tuple[ObservationPresetInfo, ...]
    observation_resolution_bounds: ObservationResolutionBounds
    observation_source_geometries: tuple[ObservationSourceGeometryInfo, ...]
    camera_settings: tuple[SelectOption, ...]
    race_modes: tuple[SelectOption, ...]
    gp_difficulties: tuple[SelectOption, ...]
    track_sampling_modes: tuple[SelectOption, ...]
    track_cups: tuple[TrackCupInfo, ...]
    built_in_courses: tuple[BuiltInCourseInfo, ...]
    vehicles: tuple[VehicleInfo, ...]
    steering_modes: tuple[SelectOption, ...]
    drive_modes: tuple[SelectOption, ...]
    lean_output_modes: tuple[SelectOption, ...]
    lean_modes: tuple[SelectOption, ...]
    stack_modes: tuple[SelectOption, ...]
    resize_filters: tuple[SelectOption, ...]
    progress_sources: tuple[SelectOption, ...]
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
    dropout_prob: float


class ConvLayerPreview(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    kind: CnnLayerKind
    in_channels: int
    out_channels: int
    kernel_size: int
    stride: int
    padding: int
    post_activation: bool
    activation: CnnActivationName | None = None
    input_height: int
    input_width: int
    output_height: int
    output_width: int
    dropped_height: int
    dropped_width: int
    params: int


class ParameterGroupPreview(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    params: int


class ActionBranchPreview(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    kind: str
    size: int
    enabled: bool
    mask_label: str | None = None


class ArchitectureNodePreview(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    label: str
    detail: str
    params: int | None = None
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
    action_branches: tuple[ActionBranchPreview, ...]
    continuous_action_dims: int
    discrete_action_logits: int
    parameter_groups: tuple[ParameterGroupPreview, ...]
    total_params: int
    architecture_lanes: tuple[ArchitectureLanePreview, ...]
