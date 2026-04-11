# src/rl_fzerox/core/config/schema.py
from __future__ import annotations

from pathlib import Path
from typing import Literal

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


class ActionMaskConfig(BaseModel):
    """Optional branch-wise action-mask restrictions for MultiDiscrete actions."""

    model_config = ConfigDict(extra="forbid")

    steer: tuple[NonNegativeInt, ...] | None = None
    drive: tuple[NonNegativeInt, ...] | None = None
    boost: tuple[NonNegativeInt, ...] | None = None
    shoulder: tuple[NonNegativeInt, ...] | None = None

    @field_validator("steer", "drive", "boost", "shoulder")
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
        for branch_name in ("steer", "drive", "boost", "shoulder"):
            values = getattr(self, branch_name)
            if values is not None:
                overrides[branch_name] = tuple(int(value) for value in values)
        return overrides


class ActionConfig(BaseModel):
    """Policy action adapter settings for the current env."""

    model_config = ConfigDict(extra="forbid")

    name: Literal[
        "steer_drive",
        "steer_drive_boost",
        "steer_drive_boost_drift",
    ] = "steer_drive_boost_drift"
    steer_buckets: int = Field(default=7, ge=3)
    mask: ActionMaskConfig | None = None

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
    preset: Literal["native_crop_v1", "native_crop_v2", "native_crop_v3"] = "native_crop_v3"
    frame_stack: PositiveInt = 4


class EnvConfig(BaseModel):
    """Environment-level rollout settings that affect frame stepping."""

    model_config = ConfigDict(extra="forbid")

    action_repeat: PositiveInt = 3
    # The step-like env limits below are counted per internal telemetry sample,
    # i.e. once per emulated frame, not once per outer env.step().
    max_episode_steps: PositiveInt = 12_000
    stuck_step_limit: PositiveInt = 240
    stuck_min_speed_kph: NonNegativeFloat = 50.0
    wrong_way_timer_limit: PositiveInt = 300
    progress_frontier_stall_limit_frames: PositiveInt | None = 900
    progress_frontier_epsilon: NonNegativeFloat = 100.0
    boost_min_energy_fraction: float = Field(default=0.1, ge=0.0, le=1.0)
    terminate_on_energy_depleted: bool = True
    randomize_game_rng_on_reset: bool = False
    randomize_game_rng_requires_race_mode: bool = True
    reset_to_race: bool = False
    action: ActionConfig = Field(default_factory=ActionConfig)
    observation: ObservationConfig = Field(default_factory=ObservationConfig)


class RewardConfig(BaseModel):
    """Reward-shaping settings for the current env."""

    model_config = ConfigDict(extra="forbid")

    name: Literal["race_v2"] = "race_v2"
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
    bootstrap_position_multiplier_scale: NonNegativeFloat = 0.0
    bootstrap_lap_count: PositiveInt = 1
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
    airborne_landing_reward: float = 0.0
    boost_redundant_press_penalty: float = 0.0
    collision_recoil_penalty: float = -2.0
    spinning_out_penalty: float = -4.0
    terminal_failure_base_penalty: float = -120.0
    stuck_truncation_base_penalty: float = -150.0
    wrong_way_truncation_base_penalty: float = -170.0
    progress_stalled_truncation_base_penalty: float = -150.0
    timeout_truncation_base_penalty: float = -150.0
    finish_position_scale: NonNegativeFloat = 4.0


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
    fps: PositiveFloat | Literal["auto"] | None = "auto"
    deterministic_policy: bool = True
    device: Literal["auto", "cpu", "cuda"] = "cpu"
    policy_run_dir: Path | None = None
    policy_artifact: Literal["latest", "best", "final"] = "latest"


class NetArchConfig(BaseModel):
    """SB3 actor/critic head sizes after the shared CNN extractor."""

    model_config = ConfigDict(extra="forbid")

    pi: tuple[PositiveInt, ...] = (256, 256)
    vf: tuple[PositiveInt, ...] = (256, 256)


class ExtractorConfig(BaseModel):
    """Shared feature-extractor settings for the PPO policy."""

    model_config = ConfigDict(extra="forbid")

    features_dim: PositiveInt | Literal["auto"] = 512
    state_features_dim: PositiveInt = 64


class PolicyRecurrentConfig(BaseModel):
    """Optional LSTM settings used only by recurrent PPO-family policies."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    hidden_size: PositiveInt = 512
    n_lstm_layers: PositiveInt = 1
    shared_lstm: bool = False
    enable_critic_lstm: bool = True


class PolicyConfig(BaseModel):
    """PPO policy and feature-extractor sizes."""

    model_config = ConfigDict(extra="forbid")

    extractor: ExtractorConfig = Field(default_factory=ExtractorConfig)
    recurrent: PolicyRecurrentConfig = Field(default_factory=PolicyRecurrentConfig)
    activation: Literal["tanh", "relu"] = "tanh"
    net_arch: NetArchConfig = Field(default_factory=NetArchConfig)


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
    """One curriculum stage with optional promotion trigger and mask override."""

    model_config = ConfigDict(extra="forbid")

    name: str
    until: CurriculumTriggerConfig | None = None
    action_mask: ActionMaskConfig | None = None


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
    """PPO training settings for the current run."""

    model_config = ConfigDict(extra="forbid")

    algorithm: Literal[
        "auto",
        "ppo",
        "maskable_ppo",
        "maskable_recurrent_ppo",
    ] = "maskable_ppo"
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
    ent_coef: NonNegativeFloat = 0.01
    vf_coef: PositiveFloat = 0.5
    max_grad_norm: PositiveFloat = 0.5
    verbose: int = Field(default=0, ge=0, le=2)
    device: str = "auto"
    save_freq: PositiveInt = 1_000
    output_root: Path = Path("local/runs")
    run_name: str = "ppo_cnn"
    init_run_dir: Path | None = None
    init_artifact: Literal["latest", "best", "final"] = "latest"


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
        if recurrent_enabled and algorithm != "maskable_recurrent_ppo":
            raise ValueError(
                "policy.recurrent.enabled=true requires "
                "train.algorithm=maskable_recurrent_ppo"
            )
        if not recurrent_enabled and algorithm == "maskable_recurrent_ppo":
            raise ValueError(
                "train.algorithm=maskable_recurrent_ppo requires "
                "policy.recurrent.enabled=true"
            )
        return self
