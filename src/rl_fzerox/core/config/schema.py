# src/rl_fzerox/core/config/schema.py
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Literal, TypeAlias

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    FilePath,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    ValidationInfo,
    field_validator,
    model_validator,
)

from rl_fzerox.core.config.action_branches import (
    ActionBranchesConfig,
    compile_action_branches,
)
from rl_fzerox.core.domain.action_adapters import DEFAULT_ACTION_ADAPTER_NAME, ActionAdapterName
from rl_fzerox.core.domain.action_values import (
    ActionMaskValue,
    compile_action_mask_values,
)
from rl_fzerox.core.domain.camera import CameraSettingName
from rl_fzerox.core.domain.lean import DEFAULT_LEAN_MODE, LeanMode
from rl_fzerox.core.domain.training_algorithms import (
    DEFAULT_TRAIN_ALGORITHM,
    RECURRENT_TRAINING_ALGORITHMS,
    TRAIN_ALGORITHM_SAC,
    TrainAlgorithmName,
)

WatchFpsSetting: TypeAlias = PositiveFloat | Literal["auto", "unlimited"]
ActionHistoryControlName: TypeAlias = Literal[
    "steer",
    "gas",
    "thrust",
    "air_brake",
    "boost",
    "lean",
]
TrackSamplingMode: TypeAlias = Literal["random", "balanced"]
ObservationCourseContext: TypeAlias = Literal["none", "one_hot_builtin"]
ObservationGroundEffectContext: TypeAlias = Literal["none", "effect_flags"]
ObservationStateComponentName: TypeAlias = Literal[
    "vehicle_state",
    "track_position",
    "surface_state",
    "course_context",
    "legacy_state",
    "control_history",
]


class ActionMaskConfig(BaseModel):
    """Optional branch-wise action-mask restrictions for MultiDiscrete actions."""

    model_config = ConfigDict(extra="forbid")

    steer: tuple[ActionMaskValue, ...] | None = None
    drive: tuple[ActionMaskValue, ...] | None = None
    gas: tuple[ActionMaskValue, ...] | None = None
    air_brake: tuple[ActionMaskValue, ...] | None = None
    boost: tuple[ActionMaskValue, ...] | None = None
    lean: tuple[ActionMaskValue, ...] | None = None

    @field_validator("steer", "drive", "gas", "air_brake", "boost", "lean")
    @classmethod
    def _validate_non_empty_mask_branch(
        cls,
        value: tuple[ActionMaskValue, ...] | None,
        info: ValidationInfo,
    ) -> tuple[ActionMaskValue, ...] | None:
        if value is not None and len(value) == 0:
            raise ValueError("Action mask branches must not be empty")
        if value is not None:
            compile_action_mask_values(str(info.field_name), value)
        return value

    def branch_overrides(self) -> dict[str, tuple[int, ...]]:
        """Return the explicitly configured branch restrictions only."""

        overrides: dict[str, tuple[int, ...]] = {}
        for branch_name in ("steer", "drive", "gas", "air_brake", "boost", "lean"):
            values = getattr(self, branch_name)
            if values is not None:
                overrides[branch_name] = compile_action_mask_values(branch_name, values)
        return overrides


class ActionConfig(BaseModel):
    """Policy action adapter settings for the current env."""

    model_config = ConfigDict(extra="forbid")

    name: ActionAdapterName = DEFAULT_ACTION_ADAPTER_NAME
    steer_buckets: int = Field(default=7, ge=3)
    steer_response_power: PositiveFloat = 1.0
    continuous_drive_mode: Literal["threshold", "pwm", "always_accelerate"] = "threshold"
    continuous_drive_deadzone: float = Field(default=0.2, ge=0.0, lt=1.0)
    continuous_air_brake_mode: Literal["always", "disable_on_ground", "off"] = "always"
    continuous_lean_deadzone: float = Field(default=0.333333, ge=0.0, lt=1.0)
    lean_mode: LeanMode = DEFAULT_LEAN_MODE
    boost_unmask_max_speed_kph: NonNegativeFloat | None = None
    boost_decision_interval_frames: PositiveInt = 1
    boost_request_lockout_frames: NonNegativeInt = 0
    lean_unmask_min_speed_kph: NonNegativeFloat | None = None
    mask: ActionMaskConfig | None = None
    branches: ActionBranchesConfig | None = None

    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy_action_fields(cls, data: object) -> object:
        # COMPAT SHIM: legacy action config field names.
        # Early run manifests used `boost_unmask_min_speed_kph`, but the intended
        # boost gate is a max-speed cap: allow boost while slower, mask it when
        # already fast. Older air-brake booleans existed before
        # `continuous_air_brake_mode`.
        if not isinstance(data, Mapping):
            return data
        values: dict[str, object] = {str(key): value for key, value in data.items()}
        missing = object()
        legacy_gate = values.pop("boost_unmask_min_speed_kph", missing)
        if legacy_gate is not missing and "boost_unmask_max_speed_kph" not in values:
            values["boost_unmask_max_speed_kph"] = legacy_gate

        legacy_air_brake_enabled = values.pop("continuous_air_brake_enabled", missing)
        legacy_disable_on_ground = values.pop(
            "continuous_air_brake_disable_on_ground",
            missing,
        )
        legacy_airborne_only = values.pop("continuous_air_brake_airborne_only", missing)
        if "continuous_air_brake_mode" not in values:
            if legacy_air_brake_enabled is False:
                values["continuous_air_brake_mode"] = "off"
            elif legacy_disable_on_ground is True or legacy_airborne_only is True:
                values["continuous_air_brake_mode"] = "disable_on_ground"

        branch_config = values.get("branches")
        if branch_config is not None:
            # COMPAT SHIM: branch-style configs are the source of truth, but
            # the env runtime still consumes adapter-era fields. Ignore old
            # action fields merged from base YAMLs or saved manifests here.
            # Delete this compile bridge when the runtime consumes branches
            # directly.
            compiled = compile_action_branches(branch_config)
            preserved_branch_fields = {
                field_name: values[field_name]
                for field_name in (
                    "continuous_drive_mode",
                    "continuous_drive_deadzone",
                )
                if field_name in values
            }
            values = {"branches": branch_config, **preserved_branch_fields}
            values.update(compiled.values)
        return values

    @field_validator("steer_buckets")
    @classmethod
    def _validate_odd_steer_buckets(cls, value: int) -> int:
        if value % 2 == 0:
            raise ValueError("steer_buckets must be odd so one bucket maps to straight")
        return value


class ObservationStateComponentConfig(BaseModel):
    """One ordered scalar-state component in the image-state observation."""

    model_config = ConfigDict(extra="forbid")

    name: ObservationStateComponentName
    speed_normalizer_kph: PositiveFloat | None = None
    lateral_velocity_normalizer: PositiveFloat | None = None
    sliding_lateral_velocity_threshold: PositiveFloat | None = None
    encoding: ObservationCourseContext | None = None
    state_profile: Literal[
        "default",
        "steer_history",
        "race_core",
    ] | None = None
    length: PositiveInt | None = Field(default=None, le=16)
    controls: tuple[ActionHistoryControlName, ...] | None = None

    @model_validator(mode="before")
    @classmethod
    def _parse_lego_component(cls, data: object) -> object:
        if isinstance(data, str):
            return {"name": data}
        if not isinstance(data, Mapping) or "name" in data or len(data) != 1:
            return data

        name, settings = next(iter(data.items()))
        if settings is None:
            return {"name": name}
        if isinstance(settings, Mapping):
            return {"name": name, **settings}
        return data

    @model_validator(mode="after")
    def _validate_component_settings(self) -> ObservationStateComponentConfig:
        configured_fields = {
            name
            for name in (
                "speed_normalizer_kph",
                "lateral_velocity_normalizer",
                "sliding_lateral_velocity_threshold",
                "encoding",
                "state_profile",
                "length",
                "controls",
            )
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
            case "vehicle_state":
                return frozenset(
                    {
                        "speed_normalizer_kph",
                        "lateral_velocity_normalizer",
                        "sliding_lateral_velocity_threshold",
                    }
                )
            case "course_context":
                return frozenset({"encoding"})
            case "legacy_state":
                return frozenset({"state_profile"})
            case "control_history":
                return frozenset({"length", "controls"})
            case "track_position" | "surface_state":
                return frozenset()

    def data(self) -> dict[str, object]:
        """Return the compact ordered form consumed by env code."""

        return self.model_dump(mode="python", exclude_none=True)


class ObservationConfig(BaseModel):
    """Observation adapter settings for the current env."""

    model_config = ConfigDict(extra="forbid")

    mode: Literal["image", "image_state"] = "image"
    state_profile: Literal[
        "default",
        "steer_history",
        "race_core",
    ] = "default"
    preset: Literal[
        "native_crop_v1",
        "native_crop_v2",
        "native_crop_v3",
        "native_crop_v4",
        "native_crop_v6",
    ] = "native_crop_v3"
    frame_stack: PositiveInt = 4
    stack_mode: Literal["rgb", "rgb_gray"] = "rgb"
    course_context: ObservationCourseContext = "none"
    ground_effect_context: ObservationGroundEffectContext = "none"
    action_history_len: PositiveInt | None = Field(default=None, le=16)
    action_history_controls: tuple[ActionHistoryControlName, ...] = (
        "steer",
        "gas",
        "boost",
        "lean",
    )
    state_components: tuple[ObservationStateComponentConfig, ...] | None = None

    @field_validator("action_history_controls")
    @classmethod
    def _validate_unique_action_history_controls(
        cls,
        value: tuple[ActionHistoryControlName, ...],
    ) -> tuple[ActionHistoryControlName, ...]:
        if len(set(value)) != len(value):
            raise ValueError("action_history_controls must not contain duplicates")
        normalized = {"gas" if control == "thrust" else control for control in value}
        if len(normalized) != len(value):
            raise ValueError("action_history_controls cannot contain both gas and thrust")
        return value

    @field_validator("state_components")
    @classmethod
    def _validate_unique_state_components(
        cls,
        value: tuple[ObservationStateComponentConfig, ...] | None,
    ) -> tuple[ObservationStateComponentConfig, ...] | None:
        if value is None:
            return None
        names = [component.name for component in value]
        if len(set(names)) != len(names):
            raise ValueError("observation.state_components must not contain duplicates")
        return value

    def state_components_data(self) -> tuple[dict[str, object], ...] | None:
        """Return state-component settings in the plain form consumed by env code."""

        if self.state_components is None:
            return None
        return tuple(component.data() for component in self.state_components)


class TrackRecordEntryConfig(BaseModel):
    """One human reference time for a track."""

    model_config = ConfigDict(extra="forbid")

    time_ms: PositiveInt
    player: str | None = None
    date: str | None = None
    mode: Literal["NTSC", "PAL"] | None = None

    def info(self, prefix: str) -> dict[str, object]:
        """Return flat, pickle-safe info fields for HUD/runtime payloads."""

        info: dict[str, object] = {f"{prefix}_time_ms": int(self.time_ms)}
        if self.player is not None:
            info[f"{prefix}_player"] = self.player
        if self.date is not None:
            info[f"{prefix}_date"] = self.date
        if self.mode is not None:
            info[f"{prefix}_mode"] = self.mode
        return info


class TrackRecordsConfig(BaseModel):
    """External reference records for a track."""

    model_config = ConfigDict(extra="forbid")

    source_label: str = "F-Zero X WR History"
    source_url: str | None = None
    non_agg_best: TrackRecordEntryConfig | None = None
    non_agg_worst: TrackRecordEntryConfig | None = None

    def info(self) -> dict[str, object]:
        """Return flat, pickle-safe info fields for HUD/runtime payloads."""

        info: dict[str, object] = {"track_record_source_label": self.source_label}
        if self.source_url is not None:
            info["track_record_source_url"] = self.source_url
        if self.non_agg_best is not None:
            info.update(self.non_agg_best.info("track_non_agg_best"))
        if self.non_agg_worst is not None:
            info.update(self.non_agg_worst.info("track_non_agg_worst"))
        return info


class TrackSamplingEntryConfig(BaseModel):
    """One reset-time baseline candidate for multi-track training."""

    model_config = ConfigDict(extra="forbid")

    id: str
    display_name: str | None = None
    course_ref: str | None = None
    course_id: str | None = None
    course_name: str | None = None
    baseline_state_path: Path
    weight: PositiveFloat = 1.0
    course_index: NonNegativeInt | None = None
    mode: str | None = None
    vehicle: str | None = None
    vehicle_name: str | None = None
    engine_setting: str | None = None
    records: TrackRecordsConfig | None = None


class TrackSamplingConfig(BaseModel):
    """Optional weighted baseline sampling performed at episode reset."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    sampling_mode: TrackSamplingMode = "random"
    entries: tuple[TrackSamplingEntryConfig, ...] = ()

    @model_validator(mode="after")
    def _validate_entries_when_enabled(self) -> TrackSamplingConfig:
        if self.enabled and not self.entries:
            raise ValueError("env.track_sampling.entries must not be empty when enabled")
        return self


class EnvConfig(BaseModel):
    """Environment-level rollout settings that affect frame stepping."""

    model_config = ConfigDict(extra="forbid")

    action_repeat: PositiveInt = 3
    # The step-like env limits below are counted per internal telemetry sample,
    # i.e. once per emulated frame, not once per outer env.step().
    max_episode_steps: PositiveInt = 12_000
    stuck_truncation_enabled: bool = True
    stuck_step_limit: PositiveInt = 240
    stuck_min_speed_kph: NonNegativeFloat = 50.0
    wrong_way_truncation_enabled: bool = True
    wrong_way_timer_limit: PositiveInt = 300
    progress_frontier_stall_limit_frames: PositiveInt | None = 900
    progress_frontier_epsilon: NonNegativeFloat = 100.0
    boost_min_energy_fraction: float = Field(default=0.1, ge=0.0, le=1.0)
    terminate_on_energy_depleted: bool = True
    randomize_game_rng_on_reset: bool = False
    randomize_game_rng_requires_race_mode: bool = True
    camera_setting: CameraSettingName | None = None
    reset_to_race: bool = False
    race_intro_target_timer: int | None = Field(default=39, ge=0, le=460)
    cache_track_baselines: bool = True
    track_sampling: TrackSamplingConfig = Field(default_factory=TrackSamplingConfig)
    action: ActionConfig = Field(default_factory=ActionConfig)
    observation: ObservationConfig = Field(default_factory=ObservationConfig)


class RewardConfig(BaseModel):
    """Reward-shaping settings for the current env."""

    model_config = ConfigDict(extra="forbid")

    name: Literal["race_v3"] = "race_v3"
    time_penalty_per_frame: float = -0.005
    reverse_time_penalty_scale: NonNegativeFloat = 2.0
    low_speed_time_penalty_scale: NonNegativeFloat = 2.0
    progress_bucket_distance: PositiveFloat = 1_000.0
    progress_bucket_reward: NonNegativeFloat = 1.0
    progress_reward_interval_frames: PositiveInt = 1
    lap_completion_bonus: NonNegativeFloat = 5.0
    lap_position_scale: NonNegativeFloat = 1.0
    energy_loss_epsilon: NonNegativeFloat = 0.01
    energy_refill_progress_multiplier: float = Field(default=1.0, ge=1.0)
    dirt_progress_multiplier: float = Field(default=1.0, ge=0.0)
    ice_progress_multiplier: float = Field(default=1.0, ge=0.0)
    energy_refill_collision_cooldown_frames: NonNegativeInt = 0
    energy_full_refill_lap_bonus: NonNegativeFloat = 0.0
    energy_full_refill_min_gain_fraction: float = Field(default=0.0, ge=0.0, le=1.0)
    gas_underuse_penalty: float = Field(default=0.0, le=0.0)
    gas_underuse_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    steer_oscillation_penalty: float = Field(default=0.0, le=0.0)
    steer_oscillation_deadzone: NonNegativeFloat = 0.0
    steer_oscillation_cap: PositiveFloat = 2.0
    steer_oscillation_power: PositiveFloat = 2.0
    lean_low_speed_penalty: float = Field(default=0.0, le=0.0)
    lean_low_speed_penalty_max_speed_kph: NonNegativeFloat = 800.0
    damage_taken_frame_penalty: float = Field(default=0.0, le=0.0)
    damage_taken_streak_ramp_penalty: float = Field(default=0.0, le=0.0)
    damage_taken_streak_cap_frames: NonNegativeInt = 0
    airborne_landing_reward: float = 0.0
    boost_pad_reward: NonNegativeFloat = 0.0
    boost_pad_reward_progress_window: PositiveFloat = 1_000.0
    collision_recoil_penalty: float = -2.0
    failure_penalty: float = -20.0
    truncation_penalty: float = -20.0


class EmulatorConfig(BaseModel):
    """Paths used to boot the libretro core, content, and optional state."""

    model_config = ConfigDict(extra="forbid")

    core_path: FilePath
    rom_path: FilePath
    runtime_dir: Path | None = None
    baseline_state_path: Path | None = None
    renderer: Literal["angrylion", "gliden64"] = "angrylion"


class TrackConfig(BaseModel):
    """Metadata for a concrete track/mode baseline."""

    model_config = ConfigDict(extra="forbid")

    id: str | None = None
    display_name: str | None = None
    course_ref: str | None = None
    course_id: str | None = None
    course_name: str | None = None
    course_index: NonNegativeInt | None = None
    mode: str | None = None
    vehicle: str | None = None
    vehicle_name: str | None = None
    engine_setting: str | None = None
    baseline_state_path: Path | None = None
    records: TrackRecordsConfig | None = None
    notes: str | None = None


class WatchConfig(BaseModel):
    """Human-facing watch UI settings."""

    model_config = ConfigDict(extra="forbid")

    episodes: PositiveInt | None = None
    fps: WatchFpsSetting | None = None
    control_fps: WatchFpsSetting | None = None
    render_fps: WatchFpsSetting | None = None
    deterministic_policy: bool = True
    device: Literal["auto", "cpu", "cuda"] = "cpu"
    policy_run_dir: Path | None = None
    policy_artifact: Literal["latest", "best", "final"] = "latest"

    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy_fps(cls, data: object) -> object:
        # COMPAT SHIM: legacy watch FPS field.
        # `fps` used to control both stepping and rendering; split it into
        # `control_fps` and `render_fps` while keeping old local configs valid.
        if not isinstance(data, Mapping):
            return data
        values: dict[str, object] = {str(key): value for key, value in data.items()}
        legacy_fps = values.get("fps")
        if legacy_fps is not None:
            values.setdefault("control_fps", legacy_fps)
            values.setdefault("render_fps", legacy_fps)
        return values

    @model_validator(mode="after")
    def _default_split_fps(self) -> WatchConfig:
        if self.control_fps is None:
            self.control_fps = "auto"
        if self.render_fps is None:
            self.render_fps = 60.0
        return self


class NetArchConfig(BaseModel):
    """SB3 actor/critic head sizes after the shared CNN extractor."""

    model_config = ConfigDict(extra="forbid")

    pi: tuple[PositiveInt, ...] = (256, 256)
    vf: tuple[PositiveInt, ...] = (256, 256)


class ExtractorConfig(BaseModel):
    """Shared feature-extractor settings for SB3 policies."""

    model_config = ConfigDict(extra="forbid")

    conv_profile: Literal["auto", "nature", "compact_deep", "compact_bottleneck"] = "auto"
    features_dim: PositiveInt | Literal["auto"] = 512
    state_features_dim: PositiveInt = 64
    fusion_features_dim: PositiveInt | None = None


class PolicyRecurrentConfig(BaseModel):
    """Optional LSTM settings used only by recurrent PPO-family policies."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    hidden_size: PositiveInt = 512
    n_lstm_layers: PositiveInt = 1
    shared_lstm: bool = False
    enable_critic_lstm: bool = True


class PolicyConfig(BaseModel):
    """SB3 policy and feature-extractor sizes."""

    model_config = ConfigDict(extra="forbid")

    extractor: ExtractorConfig = Field(default_factory=ExtractorConfig)
    recurrent: PolicyRecurrentConfig = Field(default_factory=PolicyRecurrentConfig)
    activation: Literal["tanh", "relu"] = "tanh"
    net_arch: NetArchConfig = Field(default_factory=NetArchConfig)


class CurriculumTrainOverridesConfig(BaseModel):
    """Training hyperparameter overrides applied while one stage is active."""

    model_config = ConfigDict(extra="forbid")

    learning_rate: PositiveFloat | None = None
    n_epochs: PositiveInt | None = None
    batch_size: PositiveInt | None = None
    clip_range: PositiveFloat | None = None
    ent_coef: NonNegativeFloat | None = None


class CurriculumTriggerConfig(BaseModel):
    """Episode-smoothed promotion condition for one curriculum stage."""

    model_config = ConfigDict(extra="forbid")

    race_laps_completed_mean_gte: NonNegativeFloat | None = None

    @model_validator(mode="after")
    def _validate_exactly_one_trigger(self) -> CurriculumTriggerConfig:
        if self.race_laps_completed_mean_gte is None:
            raise ValueError("Curriculum stage triggers must set race_laps_completed_mean_gte")
        return self


class CurriculumStageConfig(BaseModel):
    """One curriculum stage with optional promotion trigger and overrides."""

    model_config = ConfigDict(extra="forbid")

    name: str
    until: CurriculumTriggerConfig | None = None
    action_mask: ActionMaskConfig | None = None
    lean_unmask_min_speed_kph: NonNegativeFloat | None = None
    track_sampling: TrackSamplingConfig | None = None
    train: CurriculumTrainOverridesConfig | None = None


class CurriculumConfig(BaseModel):
    """Optional stage-based action-mask curriculum for training."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    smoothing_episodes: PositiveInt = 1
    min_stage_episodes: PositiveInt = 1
    stages: tuple[CurriculumStageConfig, ...] = ()

    @model_validator(mode="after")
    def _validate_enabled_stages(self) -> CurriculumConfig:
        if self.enabled and len(self.stages) == 0:
            raise ValueError("Enabled curriculum requires at least one stage")
        return self


class TrainConfig(BaseModel):
    """Training settings for the current run."""

    model_config = ConfigDict(extra="forbid")

    algorithm: TrainAlgorithmName = DEFAULT_TRAIN_ALGORITHM
    vec_env: Literal["dummy", "subproc"] = "dummy"
    num_envs: PositiveInt = 1
    total_timesteps: PositiveInt = 1_000_000
    n_steps: PositiveInt = 1_024
    n_epochs: PositiveInt = 10
    batch_size: PositiveInt = 256
    learning_rate: PositiveFloat = 3e-4
    gamma: float = Field(default=0.99, gt=0.0, le=1.0)
    gae_lambda: float = Field(default=0.95, gt=0.0, le=1.0)
    clip_range: PositiveFloat = 0.2
    ent_coef: NonNegativeFloat | Literal["auto"] = 0.01
    vf_coef: PositiveFloat = 0.5
    max_grad_norm: PositiveFloat = 0.5
    buffer_size: PositiveInt = 1_000_000
    learning_starts: NonNegativeInt = 100
    tau: PositiveFloat = Field(default=0.005, le=1.0)
    train_freq: PositiveInt = 1
    gradient_steps: PositiveInt = 1
    target_update_interval: PositiveInt = 1
    target_entropy: float | Literal["auto"] = "auto"
    optimize_memory_usage: bool = False
    verbose: int = Field(default=0, ge=0, le=2)
    device: str = "auto"
    save_freq: PositiveInt = 1_000
    output_root: Path = Path("local/runs")
    run_name: str = "ppo_cnn"
    init_run_dir: Path | None = None
    init_artifact: Literal["latest", "best", "final"] = "latest"

    @model_validator(mode="after")
    def _validate_algorithm_specific_values(self) -> TrainConfig:
        if self.ent_coef == "auto" and self.algorithm != TRAIN_ALGORITHM_SAC:
            raise ValueError("train.ent_coef=auto is only supported with train.algorithm=sac")
        return self


class WatchAppConfig(BaseModel):
    """Top-level watch application configuration."""

    model_config = ConfigDict(extra="forbid")

    seed: int | None = None
    emulator: EmulatorConfig
    track: TrackConfig = Field(default_factory=TrackConfig)
    env: EnvConfig = Field(default_factory=EnvConfig)
    reward: RewardConfig = Field(default_factory=RewardConfig)
    curriculum: CurriculumConfig = Field(default_factory=CurriculumConfig)
    watch: WatchConfig = Field(default_factory=WatchConfig)


class TrainAppConfig(BaseModel):
    """Top-level train application configuration."""

    model_config = ConfigDict(extra="forbid")

    seed: int | None = None
    emulator: EmulatorConfig
    track: TrackConfig = Field(default_factory=TrackConfig)
    env: EnvConfig = Field(default_factory=EnvConfig)
    reward: RewardConfig = Field(default_factory=RewardConfig)
    policy: PolicyConfig = Field(default_factory=PolicyConfig)
    curriculum: CurriculumConfig = Field(default_factory=CurriculumConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)

    @model_validator(mode="after")
    def _validate_recurrent_algorithm_alignment(self) -> TrainAppConfig:
        recurrent_enabled = self.policy.recurrent.enabled
        algorithm = self.train.algorithm
        if recurrent_enabled and algorithm not in RECURRENT_TRAINING_ALGORITHMS:
            raise ValueError("policy.recurrent.enabled=true requires a recurrent train.algorithm")
        if not recurrent_enabled and algorithm in RECURRENT_TRAINING_ALGORITHMS:
            raise ValueError(f"train.algorithm={algorithm} requires policy.recurrent.enabled=true")
        return self
