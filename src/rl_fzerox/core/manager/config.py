# src/rl_fzerox/core/manager/config.py
from __future__ import annotations

from typing import Literal

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
from rl_fzerox.core.manager.config_models import (
    ActionAxisMode,
    ActionDriveMode,
    ConfigVersion,
    ConvProfile,
    EngineSettingMode,
    LeanOutputMode,
    ManagedObservationConfig,
    ManagedPolicyConfig,
    ManagedRewardConfig,
    ManagedStateComponentConfig,
    ManagedStateFeatureDropoutConfig,
    ObservationPreset,
    RaceMode,
    TrackPoolMode,
    TrackSamplingMode,
    VehicleSelectionMode,
    default_state_components,
    default_state_feature_dropouts,
    managed_state_component_feature_names,
)


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

    max_episode_steps: PositiveInt = 12_000
    progress_frontier_stall_limit_frames: PositiveInt | None = 900
    progress_frontier_epsilon: NonNegativeFloat = 100.0
    renderer: RendererName = DEFAULT_RENDERER


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
        active_features = managed_state_component_feature_names(
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


def default_selected_course_ids() -> tuple[str, ...]:
    """Return the default manager course pool in game order."""

    return tuple(course.id for course in BUILT_IN_COURSES)


def built_in_course_id_set() -> frozenset[str]:
    return frozenset(course.id for course in BUILT_IN_COURSES)


__all__ = [
    "ConfigVersion",
    "ConvProfile",
    "ManagedActionConfig",
    "ManagedEnvironmentConfig",
    "ManagedObservationConfig",
    "ManagedPolicyConfig",
    "ManagedRewardConfig",
    "ManagedRunConfig",
    "ManagedStateComponentConfig",
    "ManagedStateFeatureDropoutConfig",
    "ManagedTracksConfig",
    "ManagedTrainConfig",
    "ManagedVehicleConfig",
    "ObservationPreset",
    "default_managed_run_config",
    "default_selected_course_ids",
    "default_state_components",
    "default_state_feature_dropouts",
]
