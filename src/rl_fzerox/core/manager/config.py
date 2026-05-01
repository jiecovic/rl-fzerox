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
    model_validator,
)

from rl_fzerox.core.domain.observation_components import (
    ActionHistoryControlName,
    ObservationCourseContextName,
    ObservationStateComponentName,
    ObservationStateComponentSettings,
    TrackPositionProgressSourceName,
)

ConfigVersion = Literal[1]
StackMode = Literal["rgb", "gray", "luma_chroma"]
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
StateComponentMode = Literal["include", "zero", "exclude"]
ConvProfile = Literal[
    "auto",
    "nature",
    "nature_32_64_128",
    "nature_wide",
    "nature_extra_k3",
    "compact_deep",
    "compact_bottleneck",
    "tiny_256",
]
FeatureDim: TypeAlias = PositiveInt | Literal["auto"]
ActivationName = Literal["relu", "tanh", "gelu"]


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
    course_context_dropout_prob: float = Field(default=0.0, ge=0.0, le=1.0)


class ManagedStateComponentConfig(BaseModel):
    """One state-vector component exposed by the run manager."""

    model_config = ConfigDict(extra="forbid")

    name: ObservationStateComponentName
    mode: StateComponentMode = "include"
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
            state_profile=None,
            length=None if self.length is None else int(self.length),
            controls=self.controls,
        )


class ManagedStateFeatureConfig(BaseModel):
    """Mode override for one concrete scalar state feature."""

    model_config = ConfigDict(extra="forbid")

    name: str
    mode: StateComponentMode = "include"


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
    state_feature_modes: tuple[ManagedStateFeatureConfig, ...] = Field(
        default_factory=lambda: default_state_feature_modes()
    )

    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy_observation_fields(cls, data: object) -> object:
        if not isinstance(data, Mapping):
            return data

        migrated = dict(data)
        progress_source = migrated.pop("progress_source", "segment_progress")
        zero_edge_ratio = migrated.pop("zero_edge_ratio", True)
        zero_outside_track_bounds = migrated.pop("zero_outside_track_bounds", True)
        legacy_zeroed_features = tuple(migrated.pop("zeroed_state_features", ()))
        if "state_components" not in migrated:
            migrated["state_components"] = _default_state_components_json(
                progress_source=str(progress_source)
            )
        if "state_feature_modes" not in migrated:
            zeroed_features: list[str] = []
            zeroed_features.extend(str(feature) for feature in legacy_zeroed_features)
            if zero_edge_ratio:
                zeroed_features.append("track_position.edge_ratio")
            if zero_outside_track_bounds:
                zeroed_features.append("track_position.outside_track_bounds")
            migrated["state_feature_modes"] = [
                {"name": feature, "mode": "zero"} for feature in sorted(set(zeroed_features))
            ]
        return migrated

    @model_validator(mode="after")
    def _validate_observation_components(self) -> ManagedObservationConfig:
        names = [component.name for component in self.state_components]
        if len(set(names)) != len(names):
            raise ValueError("observation.state_components must not contain duplicates")
        feature_names = [feature.name for feature in self.state_feature_modes]
        if len(set(feature_names)) != len(feature_names):
            raise ValueError("observation.state_feature_modes must not contain duplicates")
        active_features = _state_component_feature_names(self.state_components)
        unknown_features = [
            feature.name
            for feature in self.state_feature_modes
            if feature.name not in active_features
        ]
        if unknown_features:
            joined = ", ".join(sorted(unknown_features))
            raise ValueError(
                "observation.state_feature_modes must reference active state features: "
                f"{joined}"
            )
        return self


class ManagedPolicyConfig(BaseModel):
    """Policy architecture knobs exposed by the first run-manager slice."""

    model_config = ConfigDict(extra="forbid")

    conv_profile: ConvProfile = "nature_32_64_128"
    features_dim: FeatureDim = "auto"
    state_features_dim: PositiveInt = 64
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
    gas_on_logit: float = 0.5


class ManagedRewardConfig(BaseModel):
    """Reward knobs exposed by the first run-manager slice."""

    model_config = ConfigDict(extra="forbid")

    time_penalty_per_frame: float = -0.005
    reverse_time_penalty_scale: NonNegativeFloat = 2.0
    low_speed_time_penalty_scale: NonNegativeFloat = 2.0
    slow_speed_time_penalty_scale: NonNegativeFloat = 3.0
    slow_speed_time_penalty_start_kph: NonNegativeFloat = 760.0
    slow_speed_time_penalty_power: PositiveFloat = 1.0
    progress_bucket_distance: PositiveFloat = 1_000.0
    progress_bucket_reward: NonNegativeFloat = 1.0
    progress_reward_interval_frames: PositiveInt = 1
    airborne_progress_bucket_distance: PositiveFloat | None = 100.0
    outside_bounds_reentry_progress_distance_cap: NonNegativeFloat | None = 10_000.0
    airborne_offtrack_penalty_scale: NonNegativeFloat = 0.05
    airborne_offtrack_recovery_reward_scale: NonNegativeFloat = 0.05
    airborne_offtrack_recovery_requires_descending: bool = True
    airborne_offtrack_recovery_descend_epsilon: NonNegativeFloat = 1.0
    lap_completion_bonus: NonNegativeFloat = 5.0
    lap_position_scale: NonNegativeFloat = 1.0
    energy_loss_epsilon: NonNegativeFloat = 0.01
    energy_refill_progress_multiplier: float = Field(default=1.0, ge=1.0)
    dirt_progress_multiplier: float = Field(default=1.0, ge=0.0)
    ice_progress_multiplier: float = Field(default=1.0, ge=0.0)
    dirt_entry_penalty: float = Field(default=0.0, le=0.0)
    ice_entry_penalty: float = Field(default=0.0, le=0.0)
    energy_refill_collision_cooldown_frames: NonNegativeInt = 0
    energy_full_refill_lap_bonus: NonNegativeFloat = 0.0
    energy_full_refill_min_gain_fraction: float = Field(default=0.0, ge=0.0, le=1.0)
    gas_underuse_penalty: float = Field(default=0.0, le=0.0)
    gas_underuse_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    steer_oscillation_penalty: float = Field(default=0.0, le=0.0)
    steer_oscillation_deadzone: NonNegativeFloat = 0.0
    steer_oscillation_cap: PositiveFloat = 2.0
    steer_oscillation_power: PositiveFloat = 2.0
    manual_boost_reward: NonNegativeFloat = 0.01
    boost_pad_reward: NonNegativeFloat = 10.0
    boost_pad_reward_progress_window: PositiveFloat = 800.0
    lean_request_penalty: float = Field(default=-0.003, le=0.0)
    lean_low_speed_penalty: float = Field(default=-0.005, le=0.0)
    lean_low_speed_penalty_max_speed_kph: NonNegativeFloat = 750.0
    airborne_pitch_up_penalty: float = Field(default=-0.2, le=0.0)
    damage_taken_frame_penalty: float = Field(default=0.0, le=0.0)
    damage_taken_streak_ramp_penalty: float = Field(default=0.0, le=0.0)
    damage_taken_streak_cap_frames: NonNegativeInt = 0
    airborne_landing_reward: float = 0.0
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


class ManagedRunConfig(BaseModel):
    """DB-owned immutable config snapshot for one managed run."""

    model_config = ConfigDict(extra="forbid")

    version: ConfigVersion = 1
    seed: int = 123
    preset_name: str = "all-cups recurrent PPO"
    train: ManagedTrainConfig = Field(default_factory=ManagedTrainConfig)
    observation: ManagedObservationConfig = Field(default_factory=ManagedObservationConfig)
    policy: ManagedPolicyConfig = Field(default_factory=ManagedPolicyConfig)
    reward: ManagedRewardConfig = Field(default_factory=ManagedRewardConfig)


def default_managed_run_config() -> ManagedRunConfig:
    """Return the first manager preset without reading any YAML files."""

    return ManagedRunConfig()


def default_state_components() -> tuple[ManagedStateComponentConfig, ...]:
    """Return fresh state-component config objects for manager defaults."""

    return tuple(component.model_copy(deep=True) for component in DEFAULT_STATE_COMPONENTS)


def default_state_feature_modes() -> tuple[ManagedStateFeatureConfig, ...]:
    """Return feature-level state defaults that preserve the current run shape."""

    return tuple(feature.model_copy(deep=True) for feature in DEFAULT_STATE_FEATURE_MODES)


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
DEFAULT_STATE_FEATURE_MODES: tuple[ManagedStateFeatureConfig, ...] = (
    ManagedStateFeatureConfig(name="track_position.edge_ratio", mode="zero"),
    ManagedStateFeatureConfig(name="track_position.outside_track_bounds", mode="zero"),
)


def _default_state_components_json(*, progress_source: str) -> list[dict[str, object]]:
    return [
        component.model_dump(mode="json")
        | (
            {"progress_source": progress_source}
            if component.name == "track_position"
            else {}
        )
        for component in DEFAULT_STATE_COMPONENTS
    ]


def _state_component_feature_names(
    components: tuple[ManagedStateComponentConfig, ...],
) -> frozenset[str]:
    from rl_fzerox.core.envs.observations.state.components import state_component_definition

    names: set[str] = set()
    for component in components:
        settings = component.data()
        for feature in state_component_definition(settings).features(settings):
            names.add(feature.name)
    return frozenset(names)
