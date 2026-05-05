# src/rl_fzerox/core/manager/config.py
from __future__ import annotations

from collections.abc import Mapping
from typing import Literal, TypeAlias

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    field_validator,
    model_validator,
)

from rl_fzerox.core.config.renderers import DEFAULT_RENDERER, RendererName
from rl_fzerox.core.config.vehicle_catalog import known_vehicle_ids
from rl_fzerox.core.domain.courses import BUILT_IN_COURSES
from rl_fzerox.core.domain.lean import DEFAULT_LEAN_MODE, LeanMode
from rl_fzerox.core.domain.observation_components import (
    ActionHistoryControlName,
    ObservationCourseContextName,
    ObservationStateComponentName,
    ObservationStateComponentSettings,
    TrackPositionProgressSourceName,
)

ConfigVersion = Literal[1]
StackMode = Literal["rgb", "gray", "luma_chroma"]
RaceMode = Literal["time_attack", "gp_race"]
TrackSamplingMode = Literal["equal", "step_balanced"]
TrackPoolMode = Literal["built_in", "x_cup"]
VehicleSelectionMode = Literal["fixed", "pool"]
EngineSettingMode = Literal["fixed", "random_range"]
ActionAxisMode = Literal["continuous", "discrete"]
ActionDriveMode = Literal["pwm", "on_off"]
LeanOutputMode = Literal["three_way", "independent_buttons"]
ObservationPreset = Literal[
    "crop_84x116",
    "crop_92x124",
    "crop_116x164",
    "crop_98x130",
    "crop_66x82",
    "crop_60x76",
    "crop_68x68",
    "crop_84x84",
    "crop_76x100",
    "crop_64x64",
]
ObservationResizeFilter = Literal["nearest", "bilinear"]
ConvProfile = Literal[
    "auto",
    "nature",
    "nature_32_64_128",
    "nature_wide",
    "nature_extra_k3",
    "compact_deep",
    "compact_bottleneck",
    "tiny_256",
    "custom",
]
FeatureDim: TypeAlias = PositiveInt | Literal["auto"]
ActivationName = Literal["relu", "tanh", "gelu"]

_LEGACY_REMOVED_MANAGER_REWARD_FIELDS = frozenset(
    (
        "airborne_progress_bucket_distance",
        "airborne_offtrack_penalty_scale",
        "airborne_offtrack_recovery_requires_descending",
        "airborne_offtrack_recovery_descend_epsilon",
        "airborne_offtrack_recovery_reward_scale",
        "energy_full_refill_lap_bonus",
        "energy_full_refill_min_gain_fraction",
        "gas_underuse_penalty",
        "gas_underuse_threshold",
        "lean_low_speed_penalty",
        "lean_low_speed_penalty_max_speed_kph",
        "low_speed_time_penalty_scale",
        "steer_oscillation_cap",
        "steer_oscillation_deadzone",
        "steer_oscillation_penalty",
        "steer_oscillation_power",
    )
)
_LEGACY_REMOVED_MANAGER_ENVIRONMENT_FIELDS = frozenset(("terminate_on_energy_depleted",))


class ManagedTrainConfig(BaseModel):
    """Training knobs exposed by the first run-manager slice."""

    model_config = ConfigDict(extra="forbid")

    algorithm: Literal["maskable_hybrid_recurrent_ppo"] = "maskable_hybrid_recurrent_ppo"
    num_envs: PositiveInt = 10
    total_timesteps: PositiveInt = 50_000_000
    n_steps: PositiveInt = 2_048
    n_epochs: PositiveInt = 3
    batch_size: PositiveInt = 1_024
    learning_rate: PositiveFloat = 8.5e-5
    gamma: float = Field(default=0.995, gt=0.0, le=1.0)
    gae_lambda: float = Field(default=0.95, gt=0.0, le=1.0)
    clip_range: PositiveFloat = 0.19
    clip_range_vf: PositiveFloat | None = None
    ent_coef: NonNegativeFloat = 0.01
    vf_coef: PositiveFloat = 0.5
    max_grad_norm: PositiveFloat = 0.5
    normalize_advantage: bool = True
    target_kl: PositiveFloat | None = None
    stats_window_size: PositiveInt = 100
    checkpoint_every_rollouts: PositiveInt = 5
    save_latest_checkpoint: bool = True
    save_best_checkpoint: bool = True
    save_recent_checkpoints: bool = False
    recent_checkpoint_limit: PositiveInt | None = 5


class ManagedTracksConfig(BaseModel):
    """Track pool and race-mode knobs exposed by the run manager."""

    model_config = ConfigDict(extra="forbid")

    pool_mode: TrackPoolMode = "built_in"
    race_mode: RaceMode = "time_attack"
    sampling_mode: TrackSamplingMode = "step_balanced"
    selected_course_ids: tuple[str, ...] = Field(
        default_factory=lambda: tuple(course.id for course in BUILT_IN_COURSES)
    )

    @model_validator(mode="before")
    @classmethod
    def _migrate_x_cup_race_mode(cls, data: object) -> object:
        if not isinstance(data, Mapping):
            return data
        migrated = dict(data)
        if migrated.get("pool_mode") == "x_cup":
            migrated["race_mode"] = "gp_race"
        return migrated

    @model_validator(mode="after")
    def _validate_selected_course_ids(self) -> ManagedTracksConfig:
        if self.pool_mode == "x_cup" and self.race_mode != "gp_race":
            raise ValueError("tracks.pool_mode=x_cup requires tracks.race_mode=gp_race")
        if self.pool_mode == "built_in" and not self.selected_course_ids:
            raise ValueError("tracks.selected_course_ids must not be empty")
        if len(set(self.selected_course_ids)) != len(self.selected_course_ids):
            raise ValueError("tracks.selected_course_ids must not contain duplicates")
        unknown_ids = sorted(set(self.selected_course_ids) - built_in_course_id_set())
        if unknown_ids:
            joined = ", ".join(unknown_ids)
            raise ValueError(f"tracks.selected_course_ids contains unknown courses: {joined}")
        return self


class ManagedVehicleConfig(BaseModel):
    """Vehicle selection knobs exposed by the run manager."""

    model_config = ConfigDict(extra="forbid")

    selection_mode: VehicleSelectionMode = "pool"
    selected_vehicle_ids: tuple[str, ...] = Field(default_factory=lambda: ("blue_falcon",))
    engine_mode: EngineSettingMode = "fixed"
    engine_setting_raw_value: NonNegativeInt = Field(default=50, le=100)
    engine_setting_min_raw_value: NonNegativeInt = Field(default=20, le=100)
    engine_setting_max_raw_value: NonNegativeInt = Field(default=80, le=100)

    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy_vehicle_fields(cls, data: object) -> object:
        if not isinstance(data, Mapping):
            return data

        migrated = dict(data)
        if "selected_vehicle_ids" in migrated or "selection_mode" in migrated:
            return migrated

        vehicle_id = str(migrated.pop("vehicle_id", "blue_falcon"))
        raw_value = int(migrated.get("engine_setting_raw_value", 50))
        migrated["selection_mode"] = "pool"
        migrated["selected_vehicle_ids"] = [vehicle_id]
        migrated.setdefault("engine_mode", "fixed")
        migrated.setdefault("engine_setting_raw_value", raw_value)
        migrated.setdefault("engine_setting_min_raw_value", raw_value)
        migrated.setdefault("engine_setting_max_raw_value", raw_value)
        return migrated

    @model_validator(mode="after")
    def _validate_vehicle_config(self) -> ManagedVehicleConfig:
        if not self.selected_vehicle_ids:
            raise ValueError("vehicle.selected_vehicle_ids must not be empty")
        if len(set(self.selected_vehicle_ids)) != len(self.selected_vehicle_ids):
            raise ValueError("vehicle.selected_vehicle_ids must not contain duplicates")
        unknown_ids = sorted(set(self.selected_vehicle_ids) - set(known_vehicle_ids()))
        if unknown_ids:
            known = ", ".join(known_vehicle_ids())
            joined = ", ".join(unknown_ids)
            raise ValueError(
                f"vehicle.selected_vehicle_ids contains unknown vehicles: {joined}; known: {known}"
            )
        if self.selection_mode == "fixed" and len(self.selected_vehicle_ids) != 1:
            raise ValueError("vehicle.selection_mode=fixed requires exactly one selected vehicle")
        if self.engine_setting_min_raw_value > self.engine_setting_max_raw_value:
            raise ValueError(
                "vehicle.engine_setting_min_raw_value must be <= "
                "vehicle.engine_setting_max_raw_value"
            )
        return self


class ManagedActionConfig(BaseModel):
    """Action-space knobs exposed by the run manager."""

    model_config = ConfigDict(extra="forbid")

    action_repeat: PositiveInt = 2
    steering_mode: ActionAxisMode = "continuous"
    steer_buckets: int = Field(default=7, ge=3)
    drive_mode: ActionDriveMode = "on_off"
    force_full_throttle: bool = False
    continuous_drive_deadzone: float = Field(default=0.05, ge=0.0, lt=1.0)
    continuous_drive_full_threshold: float = Field(default=0.85, gt=0.0, le=1.0)
    continuous_drive_min_thrust: float = Field(default=0.25, ge=0.0, le=1.0)
    include_air_brake: bool = True
    air_brake_mode: ActionDriveMode = "on_off"
    enable_air_brake: bool = True
    mask_air_brake_on_ground: bool = False
    continuous_air_brake_deadzone: float = Field(default=0.05, ge=0.0, lt=1.0)
    continuous_air_brake_full_threshold: float = Field(default=0.85, gt=0.0, le=1.0)
    continuous_air_brake_min_duty: float = Field(default=0.0, ge=0.0, le=1.0)
    include_boost: bool = True
    enable_boost: bool = True
    boost_unmask_max_speed_kph: NonNegativeFloat | None = None
    boost_min_energy_fraction: float = Field(default=0.1, ge=0.0, le=1.0)
    include_lean: bool = True
    enable_lean: bool = True
    lean_output_mode: LeanOutputMode = "three_way"
    lean_mode: LeanMode = DEFAULT_LEAN_MODE
    lean_unmask_min_speed_kph: NonNegativeFloat | None = None
    lean_initial_lockout_frames: NonNegativeInt = 0
    include_pitch: bool = True
    enable_pitch: bool = True
    pitch_mode: ActionAxisMode = "discrete"
    pitch_buckets: int = Field(default=5, ge=3)

    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy_enable_flags(cls, data: object) -> object:
        if not isinstance(data, Mapping):
            return data
        migrated = dict(data)
        if migrated.get("drive_mode") == "always_full":
            migrated["drive_mode"] = "on_off"
            migrated["force_full_throttle"] = True
        for branch_name in ("air_brake", "boost", "lean", "pitch"):
            include_key = f"include_{branch_name}"
            enable_key = f"enable_{branch_name}"
            if enable_key not in migrated:
                migrated[enable_key] = bool(migrated.get(include_key, True))
        return migrated

    @field_validator("steer_buckets", "pitch_buckets")
    @classmethod
    def _validate_odd_bucket_count(cls, value: int) -> int:
        if value % 2 == 0:
            raise ValueError("action bucket counts must be odd")
        return value

    @model_validator(mode="after")
    def _validate_supported_layout(self) -> ManagedActionConfig:
        if self.continuous_drive_deadzone >= self.continuous_drive_full_threshold:
            raise ValueError(
                "continuous_drive_deadzone must be lower than continuous_drive_full_threshold"
            )
        if self.continuous_air_brake_deadzone >= self.continuous_air_brake_full_threshold:
            raise ValueError(
                "continuous_air_brake_deadzone must be lower than "
                "continuous_air_brake_full_threshold"
            )
        if not self.include_air_brake:
            self.enable_air_brake = False
        if not self.include_boost:
            self.enable_boost = False
        if not self.include_lean:
            self.enable_lean = False
        elif self.lean_output_mode == "independent_buttons":
            self.lean_mode = "raw"
        if not self.include_pitch:
            self.enable_pitch = False
        elif self.pitch_mode == "continuous":
            self.enable_pitch = True
        return self


class ManagedEnvironmentConfig(BaseModel):
    """Episode-limit and emulator-runtime knobs exposed by the run manager."""

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="before")
    @classmethod
    def _drop_legacy_environment_fields(cls, data: object) -> object:
        return _strip_legacy_manager_environment_fields(data)

    max_episode_steps: PositiveInt = 12_000
    progress_frontier_stall_limit_frames: PositiveInt | None = 900
    progress_frontier_epsilon: NonNegativeFloat = 100.0
    renderer: RendererName = DEFAULT_RENDERER


class ManagedStateComponentConfig(BaseModel):
    """One state-vector component exposed by the run manager."""

    model_config = ConfigDict(extra="forbid")

    name: ObservationStateComponentName
    encoding: ObservationCourseContextName | None = None
    progress_source: TrackPositionProgressSourceName | None = None
    length: PositiveInt | None = Field(default=None, le=16)
    controls: tuple[ActionHistoryControlName, ...] | None = None

    @model_validator(mode="after")
    def _validate_component_settings(self) -> ManagedStateComponentConfig:
        configured_fields = {
            name
            for name in ("encoding", "progress_source", "length", "controls")
            if getattr(self, name) is not None
        }
        invalid_fields = configured_fields - self._allowed_fields()
        if invalid_fields:
            joined = ", ".join(sorted(invalid_fields))
            raise ValueError(f"{self.name} does not accept setting(s): {joined}")
        if self.controls is not None:
            if len(set(self.controls)) != len(self.controls):
                raise ValueError("control_history.controls must not contain duplicates")
            normalized = {"gas" if control == "thrust" else control for control in self.controls}
            if len(normalized) != len(self.controls):
                raise ValueError("control_history.controls cannot contain both gas and thrust")
        return self

    def _allowed_fields(self) -> frozenset[str]:
        match self.name:
            case "course_context":
                return frozenset({"encoding"})
            case "control_history":
                return frozenset({"length", "controls"})
            case "track_position":
                return frozenset({"progress_source"})
            case "vehicle_state" | "machine_context" | "surface_state":
                return frozenset()
            case _:
                raise ValueError(f"Unsupported state component: {self.name!r}")

    def data(self) -> ObservationStateComponentSettings:
        return ObservationStateComponentSettings(
            name=self.name,
            encoding=self.encoding,
            progress_source=self.progress_source,
            length=None if self.length is None else int(self.length),
            controls=self.controls,
        )


class ManagedStateFeatureDropoutConfig(BaseModel):
    """Episode-scoped dropout override for one concrete scalar state feature."""

    model_config = ConfigDict(extra="forbid")

    name: str
    dropout_prob: float = Field(default=0.0, ge=0.0, le=1.0)


class ManagedObservationConfig(BaseModel):
    """Observation knobs exposed by the first run-manager slice."""

    model_config = ConfigDict(extra="forbid")

    preset: ObservationPreset = "crop_60x76"
    frame_stack: PositiveInt = Field(default=2, le=8)
    stack_mode: StackMode = "rgb"
    minimap_layer: bool = False
    resize_filter: ObservationResizeFilter = "bilinear"
    minimap_resize_filter: ObservationResizeFilter = "nearest"
    state_components: tuple[ManagedStateComponentConfig, ...] = Field(
        default_factory=lambda: default_state_components()
    )
    state_feature_dropouts: tuple[ManagedStateFeatureDropoutConfig, ...] = Field(
        default_factory=lambda: default_state_feature_dropouts()
    )

    @model_validator(mode="after")
    def _validate_observation_components(self) -> ManagedObservationConfig:
        names = [component.name for component in self.state_components]
        if len(set(names)) != len(names):
            raise ValueError("observation.state_components must not contain duplicates")
        feature_names = [feature.name for feature in self.state_feature_dropouts]
        if len(set(feature_names)) != len(feature_names):
            raise ValueError("observation.state_feature_dropouts must not contain duplicates")
        return self


class ManagedPolicyConfig(BaseModel):
    """Policy architecture knobs exposed by the first run-manager slice."""

    model_config = ConfigDict(extra="forbid")

    class CustomConvLayer(BaseModel):
        model_config = ConfigDict(extra="forbid")

        out_channels: PositiveInt
        kernel_size: PositiveInt
        stride: PositiveInt
        padding: NonNegativeInt = 0

    conv_profile: ConvProfile = "nature_32_64_128"
    custom_conv_layers: tuple[CustomConvLayer, ...] = Field(
        default_factory=lambda: default_custom_conv_layers()
    )
    features_dim: FeatureDim = "auto"
    state_net_arch: tuple[PositiveInt, ...] = (64,)
    fusion_features_dim: PositiveInt = 768
    layer_norm: bool = True
    activation: ActivationName = "relu"
    recurrent_enabled: bool = True
    recurrent_hidden_size: PositiveInt = 256
    recurrent_n_lstm_layers: PositiveInt = 1
    recurrent_shared_lstm: bool = False
    recurrent_enable_critic_lstm: bool = True
    pi_net_arch: tuple[PositiveInt, ...] = (256, 128)
    vf_net_arch: tuple[PositiveInt, ...] = (256, 128)
    gas_on_logit: float = 0.0

    @model_validator(mode="after")
    def _validate_custom_conv_layers(self) -> ManagedPolicyConfig:
        if self.conv_profile == "custom" and not self.custom_conv_layers:
            raise ValueError("policy.custom_conv_layers must not be empty for conv_profile=custom")
        return self


class ManagedRewardConfig(BaseModel):
    """Reward knobs exposed by the first run-manager slice."""

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="before")
    @classmethod
    def _drop_legacy_reward_fields(cls, data: object) -> object:
        return _strip_legacy_manager_reward_fields(data)

    time_penalty_per_frame: float = 0.0
    reverse_time_penalty_scale: NonNegativeFloat = 2.0
    slow_speed_time_penalty_scale: NonNegativeFloat = 3.0
    slow_speed_time_penalty_start_kph: NonNegativeFloat = 760.0
    slow_speed_time_penalty_power: PositiveFloat = 1.0
    progress_bucket_distance: PositiveFloat = 25.0
    progress_bucket_reward: NonNegativeFloat = 0.05
    progress_reward_interval_frames: PositiveInt = 1
    suspend_progress_while_outside_track_bounds: bool = True
    outside_bounds_reentry_progress_distance_cap: NonNegativeFloat | None = 10_000.0
    outside_track_frame_penalty: float = Field(default=0.0, le=0.0)
    lap_completion_bonus: NonNegativeFloat = 5.0
    lap_position_scale: NonNegativeFloat = 1.0
    energy_loss_epsilon: NonNegativeFloat = 0.01
    energy_refill_progress_multiplier: float = Field(default=3.0, ge=1.0)
    dirt_progress_multiplier: float = Field(default=1.0, ge=0.0)
    ice_progress_multiplier: float = Field(default=1.0, ge=0.0)
    dirt_entry_penalty: float = Field(default=0.0, le=0.0)
    ice_entry_penalty: float = Field(default=0.0, le=0.0)
    energy_refill_collision_cooldown_frames: NonNegativeInt = 120
    air_brake_request_penalty: float = Field(default=0.0, le=0.0)
    manual_boost_reward: NonNegativeFloat = 0.01
    boost_pad_reward: NonNegativeFloat = 10.0
    boost_pad_reward_progress_window: PositiveFloat = 800.0
    lean_request_penalty: float = Field(default=-0.003, le=0.0)
    airborne_pitch_up_penalty: float = Field(default=-0.2, le=0.0)
    damage_taken_frame_penalty: float = Field(default=-0.02, le=0.0)
    damage_taken_streak_ramp_penalty: float = Field(default=-0.001, le=0.0)
    damage_taken_streak_cap_frames: NonNegativeInt = 120
    airborne_landing_reward: float = 1.0
    collision_recoil_penalty: float = -4.0
    failure_penalty: float = -30.0
    truncation_penalty: float = -30.0
    step_reward_clip_min: float | None = -100.0
    step_reward_clip_max: float | None = 100.0

    @model_validator(mode="after")
    def _validate_step_reward_clip_bounds(self) -> ManagedRewardConfig:
        if (
            self.step_reward_clip_min is not None
            and self.step_reward_clip_max is not None
            and self.step_reward_clip_min > self.step_reward_clip_max
        ):
            raise ValueError("step_reward_clip_min must be <= step_reward_clip_max")
        return self


def _strip_legacy_manager_reward_fields(data: object) -> object:
    if not isinstance(data, Mapping):
        return data
    normalized = dict(data)
    if (
        "suspend_progress_while_outside_track_bounds" not in normalized
        and "suspend_progress_while_airborne" in normalized
    ):
        normalized["suspend_progress_while_outside_track_bounds"] = normalized[
            "suspend_progress_while_airborne"
        ]
    normalized.pop("suspend_progress_while_airborne", None)
    for field_name in _LEGACY_REMOVED_MANAGER_REWARD_FIELDS:
        normalized.pop(field_name, None)
    return normalized


def _strip_legacy_manager_environment_fields(data: object) -> object:
    if not isinstance(data, Mapping):
        return data
    normalized = dict(data)
    for field_name in _LEGACY_REMOVED_MANAGER_ENVIRONMENT_FIELDS:
        normalized.pop(field_name, None)
    return normalized


class ManagedRunConfig(BaseModel):
    """DB-owned immutable config snapshot for one managed run."""

    model_config = ConfigDict(extra="forbid")

    version: ConfigVersion = 1
    seed: int = 123
    preset_name: str = "all-cups recurrent PPO"
    tracks: ManagedTracksConfig = Field(default_factory=ManagedTracksConfig)
    vehicle: ManagedVehicleConfig = Field(default_factory=ManagedVehicleConfig)
    action: ManagedActionConfig = Field(default_factory=ManagedActionConfig)
    environment: ManagedEnvironmentConfig = Field(default_factory=ManagedEnvironmentConfig)
    train: ManagedTrainConfig = Field(default_factory=ManagedTrainConfig)
    observation: ManagedObservationConfig = Field(default_factory=ManagedObservationConfig)
    policy: ManagedPolicyConfig = Field(default_factory=ManagedPolicyConfig)
    reward: ManagedRewardConfig = Field(default_factory=ManagedRewardConfig)

    @model_validator(mode="after")
    def _validate_custom_conv_geometry(self) -> ManagedRunConfig:
        from rl_fzerox.core.manager.architecture.metadata import preset_geometry
        from rl_fzerox.core.policy.extractors import (
            ensure_conv_spec_fits_geometry,
            resolve_conv_spec,
        )

        height, width = preset_geometry(self.observation.preset)
        conv_spec = resolve_conv_spec(
            (height, width),
            conv_profile=self.policy.conv_profile,
            custom_conv_layers=tuple(
                layer.model_dump(mode="python") for layer in self.policy.custom_conv_layers
            ),
        )
        ensure_conv_spec_fits_geometry(
            height=height,
            width=width,
            conv_spec=conv_spec,
            profile_name=self.policy.conv_profile,
        )
        active_features = _state_component_feature_names(
            self.observation.state_components,
            independent_lean_buttons=self.action.lean_output_mode == "independent_buttons",
        )
        unknown_features = [
            feature.name
            for feature in self.observation.state_feature_dropouts
            if feature.name not in active_features
        ]
        if unknown_features:
            joined = ", ".join(sorted(unknown_features))
            raise ValueError(
                "observation.state_feature_dropouts must reference active state features: "
                f"{joined}"
            )
        return self


def default_managed_run_config() -> ManagedRunConfig:
    """Return the first manager preset without reading any YAML files."""

    return ManagedRunConfig()


def default_custom_conv_layers() -> tuple[ManagedPolicyConfig.CustomConvLayer, ...]:
    """Return a sensible custom-CNN starting point for the manager."""

    return (
        ManagedPolicyConfig.CustomConvLayer(out_channels=32, kernel_size=8, stride=4, padding=0),
        ManagedPolicyConfig.CustomConvLayer(out_channels=64, kernel_size=4, stride=2, padding=0),
        ManagedPolicyConfig.CustomConvLayer(out_channels=128, kernel_size=3, stride=1, padding=0),
    )


def default_selected_course_ids() -> tuple[str, ...]:
    """Return the default manager course pool in game order."""

    return tuple(course.id for course in BUILT_IN_COURSES)


def default_state_components() -> tuple[ManagedStateComponentConfig, ...]:
    """Return fresh state-component config objects for manager defaults."""

    return tuple(component.model_copy(deep=True) for component in DEFAULT_STATE_COMPONENTS)


def default_state_feature_dropouts() -> tuple[ManagedStateFeatureDropoutConfig, ...]:
    """Return feature-level state dropouts that preserve the current run shape."""

    return tuple(feature.model_copy(deep=True) for feature in DEFAULT_STATE_FEATURE_DROPOUTS)


DEFAULT_STATE_COMPONENTS: tuple[ManagedStateComponentConfig, ...] = (
    ManagedStateComponentConfig(name="vehicle_state"),
    ManagedStateComponentConfig(name="machine_context"),
    ManagedStateComponentConfig(name="track_position", progress_source="segment_progress"),
    ManagedStateComponentConfig(name="surface_state"),
    ManagedStateComponentConfig(name="course_context", encoding="one_hot_builtin"),
    ManagedStateComponentConfig(
        name="control_history",
        length=1,
        controls=("steer", "thrust", "air_brake", "boost", "lean", "pitch"),
    ),
)
DEFAULT_STATE_FEATURE_DROPOUTS: tuple[ManagedStateFeatureDropoutConfig, ...] = (
    ManagedStateFeatureDropoutConfig(name="track_position.edge_ratio", dropout_prob=1.0),
    ManagedStateFeatureDropoutConfig(
        name="track_position.outside_track_bounds",
        dropout_prob=1.0,
    ),
)


def _state_component_feature_names(
    components: tuple[ManagedStateComponentConfig, ...],
    *,
    independent_lean_buttons: bool = False,
) -> frozenset[str]:
    from rl_fzerox.core.envs.observations.state.components import state_component_features

    names: set[str] = set()
    for component in components:
        settings = component.data()
        for feature in state_component_features(
            settings,
            independent_lean_buttons=independent_lean_buttons,
        ):
            names.add(feature.name)
    return frozenset(names)


def built_in_course_id_set() -> frozenset[str]:
    return frozenset(course.id for course in BUILT_IN_COURSES)
