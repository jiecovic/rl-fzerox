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
    field_validator,
    model_validator,
)

from rl_fzerox.core.domain.action_adapters import DEFAULT_ACTION_ADAPTER_NAME, ActionAdapterName
from rl_fzerox.core.domain.camera import CameraSettingName
from rl_fzerox.core.domain.shoulder_slide import (
    DEFAULT_SHOULDER_SLIDE_MODE,
    ShoulderSlideMode,
)
from rl_fzerox.core.domain.training_algorithms import (
    DEFAULT_TRAIN_ALGORITHM,
    RECURRENT_TRAINING_ALGORITHMS,
    TRAIN_ALGORITHM_SAC,
    TrainAlgorithmName,
)

WatchFpsSetting: TypeAlias = PositiveFloat | Literal["auto", "unlimited"]


class ActionMaskConfig(BaseModel):
    """Optional branch-wise action-mask restrictions for MultiDiscrete actions."""

    model_config = ConfigDict(extra="forbid")

    steer: tuple[NonNegativeInt, ...] | None = None
    drive: tuple[NonNegativeInt, ...] | None = None
    gas: tuple[NonNegativeInt, ...] | None = None
    air_brake: tuple[NonNegativeInt, ...] | None = None
    boost: tuple[NonNegativeInt, ...] | None = None
    shoulder: tuple[NonNegativeInt, ...] | None = None

    @field_validator("steer", "drive", "gas", "air_brake", "boost", "shoulder")
    @classmethod
    def _validate_non_empty_mask_branch(
        cls,
        value: tuple[int, ...] | None,
    ) -> tuple[int, ...] | None:
        if value is not None and len(value) == 0:
            raise ValueError("Action mask branches must not be empty")
        return value

    def branch_overrides(self) -> dict[str, tuple[int, ...]]:
        """Return the explicitly configured branch restrictions only."""

        overrides: dict[str, tuple[int, ...]] = {}
        for branch_name in ("steer", "drive", "gas", "air_brake", "boost", "shoulder"):
            values = getattr(self, branch_name)
            if values is not None:
                overrides[branch_name] = tuple(int(value) for value in values)
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
    continuous_shoulder_deadzone: float = Field(default=0.333333, ge=0.0, lt=1.0)
    shoulder_slide_mode: ShoulderSlideMode = DEFAULT_SHOULDER_SLIDE_MODE
    boost_unmask_max_speed_kph: NonNegativeFloat | None = None
    boost_decision_interval_frames: PositiveInt = 1
    boost_request_lockout_frames: NonNegativeInt = 0
    shoulder_unmask_min_speed_kph: NonNegativeFloat | None = None
    mask: ActionMaskConfig | None = None

    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy_action_fields(cls, data: object) -> object:
        # COMPAT SHIM: legacy action config field names.
        # Early run manifests used `boost_unmask_min_speed_kph`, but the intended
        # boost gate is a max-speed cap: allow boost while slower, mask it when
        # already fast. They also used "drift" for the Z/R shoulder inputs and
        # older air-brake booleans before `continuous_air_brake_mode` existed.
        # Keep these saved manifests loadable until old checkpoints are retired.
        if not isinstance(data, Mapping):
            return data
        values: dict[str, object] = {str(key): value for key, value in data.items()}
        missing = object()
        legacy_gate = values.pop("boost_unmask_min_speed_kph", missing)
        if legacy_gate is not missing and "boost_unmask_max_speed_kph" not in values:
            values["boost_unmask_max_speed_kph"] = legacy_gate

        legacy_shoulder_deadzone = values.pop("continuous_drift_deadzone", missing)
        if legacy_shoulder_deadzone is not missing and "continuous_shoulder_deadzone" not in values:
            values["continuous_shoulder_deadzone"] = legacy_shoulder_deadzone

        legacy_shoulder_speed = values.pop("drift_unmask_min_speed_kph", missing)
        if legacy_shoulder_speed is not missing and "shoulder_unmask_min_speed_kph" not in values:
            values["shoulder_unmask_min_speed_kph"] = legacy_shoulder_speed

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
        return values

    @field_validator("steer_buckets")
    @classmethod
    def _validate_odd_steer_buckets(cls, value: int) -> int:
        if value % 2 == 0:
            raise ValueError("steer_buckets must be odd so one bucket maps to straight")
        return value


class ObservationConfig(BaseModel):
    """Observation adapter settings for the current env."""

    model_config = ConfigDict(extra="forbid")

    mode: Literal["image", "image_state"] = "image"
    state_profile: Literal["default", "steer_history"] = "default"
    preset: Literal["native_crop_v1", "native_crop_v2", "native_crop_v3"] = "native_crop_v3"
    frame_stack: PositiveInt = 4


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
    action: ActionConfig = Field(default_factory=ActionConfig)
    observation: ObservationConfig = Field(default_factory=ObservationConfig)


class RewardConfig(BaseModel):
    """Reward-shaping settings for the current env."""

    model_config = ConfigDict(extra="forbid")

    name: Literal["race_v2", "race_v3", "race_v4"] = "race_v2"
    time_penalty_per_frame: float = -0.005
    reverse_time_penalty_scale: NonNegativeFloat = 2.0
    low_speed_time_penalty_scale: NonNegativeFloat = 2.0
    milestone_distance: PositiveFloat = 3_000.0
    randomize_milestone_phase_on_reset: bool = False
    milestone_bonus: NonNegativeFloat = 2.0
    milestone_speed_scale: NonNegativeFloat = 0.0
    milestone_speed_bonus_cap: NonNegativeFloat = 0.0
    bootstrap_progress_scale: NonNegativeFloat = 0.001
    bootstrap_regress_penalty_scale: NonNegativeFloat = 0.002
    progress_bucket_distance: PositiveFloat = 1_000.0
    progress_bucket_reward: NonNegativeFloat = 1.0
    progress_reward_interval_frames: PositiveInt = 1
    bootstrap_position_multiplier_scale: NonNegativeFloat = 0.0
    bootstrap_lap_count: PositiveInt = 1
    lap_completion_bonus: NonNegativeFloat = 5.0
    lap_1_completion_bonus: NonNegativeFloat = 20.0
    lap_2_completion_bonus: NonNegativeFloat = 35.0
    final_lap_completion_bonus: NonNegativeFloat = 60.0
    lap_position_scale: NonNegativeFloat = 1.0
    remaining_step_penalty_per_frame: NonNegativeFloat = 0.01
    remaining_lap_penalty: NonNegativeFloat = 50.0
    energy_loss_epsilon: NonNegativeFloat = 0.01
    energy_loss_penalty_scale: NonNegativeFloat = 0.05
    energy_loss_safe_fraction: float = Field(default=0.9, ge=0.0, le=1.0)
    energy_loss_danger_power: PositiveFloat = 2.0
    energy_gain_reward_scale: NonNegativeFloat = 0.02
    energy_gain_collision_cooldown_frames: NonNegativeInt = 0
    energy_full_refill_bonus: NonNegativeFloat = 0.0
    energy_full_refill_cooldown_frames: NonNegativeInt = 0
    damage_taken_frame_penalty: float = Field(default=0.0, le=0.0)
    damage_taken_streak_ramp_penalty: float = Field(default=0.0, le=0.0)
    damage_taken_streak_cap_frames: NonNegativeInt = 0
    airborne_landing_reward: float = 0.0
    grounded_air_brake_penalty: float = 0.0
    drive_axis_negative_penalty_scale: float = Field(default=0.0, le=0.0)
    boost_pad_reward: NonNegativeFloat = 0.0
    boost_pad_reward_cooldown_frames: NonNegativeInt = 0
    boost_pad_reward_progress_window: PositiveFloat = 1_000.0
    manual_boost_request_reward: float = 0.0
    collision_recoil_penalty: float = -2.0
    spinning_out_penalty: float = -4.0
    failure_penalty: float = -20.0
    truncation_penalty: float = -20.0
    terminal_failure_base_penalty: float = -120.0
    stuck_truncation_base_penalty: float = -150.0
    wrong_way_truncation_base_penalty: float = -170.0
    progress_stalled_truncation_base_penalty: float = -150.0
    timeout_truncation_base_penalty: float = -150.0
    finish_position_scale: NonNegativeFloat = 4.0

    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy_reward_fields(cls, data: object) -> object:
        # COMPAT SHIM: legacy reward config field names.
        # Run manifests saved before `manual_boost_request_reward` still contain
        # `reward.boost_press_penalty` or the older
        # `reward.boost_redundant_press_penalty`. Watch loads those manifests to
        # reconstruct policy metadata, so reject-free migration must live at the
        # schema boundary. New YAML should only use `manual_boost_request_reward`.
        if not isinstance(data, Mapping):
            return data
        values: dict[str, object] = {str(key): value for key, value in data.items()}
        missing = object()
        legacy_redundant_penalty = values.pop("boost_redundant_press_penalty", missing)
        legacy_press_penalty = values.pop("boost_press_penalty", missing)
        if "manual_boost_request_reward" not in values:
            if legacy_press_penalty is not missing:
                values["manual_boost_request_reward"] = legacy_press_penalty
            elif legacy_redundant_penalty is not missing:
                values["manual_boost_request_reward"] = legacy_redundant_penalty
        return values


class EmulatorConfig(BaseModel):
    """Paths used to boot the libretro core, content, and optional state."""

    model_config = ConfigDict(extra="forbid")

    core_path: FilePath
    rom_path: FilePath
    runtime_dir: Path | None = None
    baseline_state_path: Path | None = None
    renderer: Literal["angrylion"] = "angrylion"


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

    ent_coef: NonNegativeFloat | None = None


class CurriculumTriggerConfig(BaseModel):
    """Episode-smoothed promotion condition for one curriculum stage."""

    model_config = ConfigDict(extra="forbid")

    race_laps_completed_mean_gte: NonNegativeFloat | None = None
    milestones_completed_mean_gte: NonNegativeFloat | None = None

    @model_validator(mode="after")
    def _validate_exactly_one_trigger(self) -> CurriculumTriggerConfig:
        configured = [
            value
            for value in (
                self.race_laps_completed_mean_gte,
                self.milestones_completed_mean_gte,
            )
            if value is not None
        ]
        if len(configured) != 1:
            raise ValueError("Curriculum stage triggers must set exactly one condition")
        return self


class CurriculumStageConfig(BaseModel):
    """One curriculum stage with optional promotion trigger and overrides."""

    model_config = ConfigDict(extra="forbid")

    name: str
    until: CurriculumTriggerConfig | None = None
    action_mask: ActionMaskConfig | None = None
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
    env: EnvConfig = Field(default_factory=EnvConfig)
    reward: RewardConfig = Field(default_factory=RewardConfig)
    curriculum: CurriculumConfig = Field(default_factory=CurriculumConfig)
    watch: WatchConfig = Field(default_factory=WatchConfig)


class TrainAppConfig(BaseModel):
    """Top-level train application configuration."""

    model_config = ConfigDict(extra="forbid")

    seed: int | None = None
    emulator: EmulatorConfig
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
