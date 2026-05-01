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
    model_validator,
)

from rl_fzerox.core.domain.observation_components import TrackPositionProgressSourceName

ConfigVersion = Literal[1]
StackMode = Literal["rgb", "gray", "luma_chroma"]
ConvProfile = Literal["nature", "nature_32_64_128", "nature_wide"]


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


class ManagedObservationConfig(BaseModel):
    """Observation knobs exposed by the first run-manager slice."""

    model_config = ConfigDict(extra="forbid")

    preset: Literal["crop_60x76"] = "crop_60x76"
    frame_stack: PositiveInt = Field(default=2, le=8)
    stack_mode: StackMode = "rgb"
    minimap_layer: bool = False
    progress_source: TrackPositionProgressSourceName = "segment_progress"
    zero_edge_ratio: bool = True
    zero_outside_track_bounds: bool = True


class ManagedPolicyConfig(BaseModel):
    """Policy architecture knobs exposed by the first run-manager slice."""

    model_config = ConfigDict(extra="forbid")

    conv_profile: ConvProfile = "nature_32_64_128"
    fusion_features_dim: PositiveInt = 768
    recurrent_hidden_size: PositiveInt = 256
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
