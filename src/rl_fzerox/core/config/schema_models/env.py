# src/rl_fzerox/core/config/schema_models/env.py
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
    model_validator,
)

from rl_fzerox.core.config.schema_models.actions import ActionConfig
from rl_fzerox.core.config.schema_models.observations import ObservationConfig
from rl_fzerox.core.config.schema_models.tracks import TrackSamplingConfig
from rl_fzerox.core.domain.camera import CameraSettingName


class EnvConfig(BaseModel):
    """Environment-level rollout settings that affect frame stepping."""

    model_config = ConfigDict(extra="forbid")

    action_repeat: PositiveInt = 3
    # The step-like env limits below are counted per internal telemetry sample,
    # i.e. once per emulated frame, not once per outer env.step().
    max_episode_steps: PositiveInt = 12_000
    stuck_min_speed_kph: NonNegativeFloat = 50.0
    progress_frontier_stall_limit_frames: PositiveInt | None = 900
    progress_frontier_epsilon: NonNegativeFloat = 100.0
    boost_min_energy_fraction: float = Field(default=0.1, ge=0.0, le=1.0)
    terminate_on_energy_depleted: bool = False
    randomize_game_rng_on_reset: bool = False
    randomize_game_rng_requires_race_mode: bool = True
    camera_setting: CameraSettingName | None = None
    reset_to_race: bool = False
    race_intro_target_timer: int | None = Field(default=39, ge=0, le=460)
    cache_track_baselines: bool = True
    track_sampling: TrackSamplingConfig = Field(default_factory=TrackSamplingConfig)
    action: ActionConfig = Field(default_factory=ActionConfig)
    observation: ObservationConfig = Field(default_factory=ObservationConfig)


class RewardCourseOverrideConfig(BaseModel):
    """Course-local reward overrides for fields in :class:`RewardConfig`."""

    model_config = ConfigDict(extra="forbid")

    time_penalty_per_frame: float | None = None
    reverse_time_penalty_scale: NonNegativeFloat | None = None
    low_speed_time_penalty_scale: NonNegativeFloat | None = None
    slow_speed_time_penalty_scale: NonNegativeFloat | None = None
    slow_speed_time_penalty_start_kph: NonNegativeFloat | None = None
    slow_speed_time_penalty_power: PositiveFloat | None = None
    progress_bucket_distance: PositiveFloat | None = None
    progress_bucket_reward: NonNegativeFloat | None = None
    progress_reward_interval_frames: PositiveInt | None = None
    airborne_progress_bucket_distance: PositiveFloat | None = None
    outside_bounds_reentry_progress_distance_cap: NonNegativeFloat | None = None
    airborne_offtrack_penalty_scale: NonNegativeFloat | None = None
    airborne_offtrack_recovery_reward_scale: NonNegativeFloat | None = None
    airborne_offtrack_recovery_requires_descending: bool | None = None
    airborne_offtrack_recovery_descend_epsilon: NonNegativeFloat | None = None
    lap_completion_bonus: NonNegativeFloat | None = None
    lap_position_scale: NonNegativeFloat | None = None
    energy_loss_epsilon: NonNegativeFloat | None = None
    energy_refill_progress_multiplier: float | None = Field(default=None, ge=1.0)
    dirt_progress_multiplier: float | None = Field(default=None, ge=0.0)
    ice_progress_multiplier: float | None = Field(default=None, ge=0.0)
    dirt_entry_penalty: float | None = Field(default=None, le=0.0)
    ice_entry_penalty: float | None = Field(default=None, le=0.0)
    energy_refill_collision_cooldown_frames: NonNegativeInt | None = None
    energy_full_refill_lap_bonus: NonNegativeFloat | None = None
    energy_full_refill_min_gain_fraction: float | None = Field(default=None, ge=0.0, le=1.0)
    gas_underuse_penalty: float | None = Field(default=None, le=0.0)
    gas_underuse_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    steer_oscillation_penalty: float | None = Field(default=None, le=0.0)
    steer_oscillation_deadzone: NonNegativeFloat | None = None
    steer_oscillation_cap: PositiveFloat | None = None
    steer_oscillation_power: PositiveFloat | None = None
    lean_request_penalty: float | None = Field(default=None, le=0.0)
    airborne_pitch_up_penalty: float | None = Field(default=None, le=0.0)
    lean_low_speed_penalty: float | None = Field(default=None, le=0.0)
    lean_low_speed_penalty_max_speed_kph: NonNegativeFloat | None = None
    damage_taken_frame_penalty: float | None = Field(default=None, le=0.0)
    damage_taken_streak_ramp_penalty: float | None = Field(default=None, le=0.0)
    damage_taken_streak_cap_frames: NonNegativeInt | None = None
    airborne_landing_reward: float | None = None
    manual_boost_reward: NonNegativeFloat | None = None
    boost_pad_reward: NonNegativeFloat | None = None
    boost_pad_reward_progress_window: PositiveFloat | None = None
    collision_recoil_penalty: float | None = None
    failure_penalty: float | None = None
    truncation_penalty: float | None = None
    step_reward_clip_min: float | None = None
    step_reward_clip_max: float | None = None

    @model_validator(mode="after")
    def _validate_step_reward_clip_bounds(self) -> RewardCourseOverrideConfig:
        _validate_step_reward_clip_bounds(
            min_value=self.step_reward_clip_min,
            max_value=self.step_reward_clip_max,
        )
        return self


class RewardConfig(BaseModel):
    """Reward-shaping settings for the current env."""

    model_config = ConfigDict(extra="forbid")

    name: Literal["race_v3"] = "race_v3"
    time_penalty_per_frame: float = -0.005
    reverse_time_penalty_scale: NonNegativeFloat = 2.0
    low_speed_time_penalty_scale: NonNegativeFloat = 2.0
    slow_speed_time_penalty_scale: NonNegativeFloat = 0.0
    slow_speed_time_penalty_start_kph: NonNegativeFloat = 0.0
    slow_speed_time_penalty_power: PositiveFloat = 1.0
    progress_bucket_distance: PositiveFloat = 1_000.0
    progress_bucket_reward: NonNegativeFloat = 1.0
    progress_reward_interval_frames: PositiveInt = 1
    airborne_progress_bucket_distance: PositiveFloat | None = None
    outside_bounds_reentry_progress_distance_cap: NonNegativeFloat | None = None
    airborne_offtrack_penalty_scale: NonNegativeFloat = 0.0
    airborne_offtrack_recovery_reward_scale: NonNegativeFloat = 0.0
    airborne_offtrack_recovery_requires_descending: bool = False
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
    lean_request_penalty: float = Field(default=0.0, le=0.0)
    airborne_pitch_up_penalty: float = Field(default=0.0, le=0.0)
    lean_low_speed_penalty: float = Field(default=0.0, le=0.0)
    lean_low_speed_penalty_max_speed_kph: NonNegativeFloat = 800.0
    damage_taken_frame_penalty: float = Field(default=0.0, le=0.0)
    damage_taken_streak_ramp_penalty: float = Field(default=0.0, le=0.0)
    damage_taken_streak_cap_frames: NonNegativeInt = 0
    airborne_landing_reward: float = 0.0
    manual_boost_reward: NonNegativeFloat = 0.0
    boost_pad_reward: NonNegativeFloat = 0.0
    boost_pad_reward_progress_window: PositiveFloat = 1_000.0
    collision_recoil_penalty: float = -2.0
    failure_penalty: float = -20.0
    truncation_penalty: float = -20.0
    step_reward_clip_min: float | None = None
    step_reward_clip_max: float | None = None
    course_overrides: dict[str, RewardCourseOverrideConfig] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_step_reward_clip_bounds(self) -> RewardConfig:
        _validate_step_reward_clip_bounds(
            min_value=self.step_reward_clip_min,
            max_value=self.step_reward_clip_max,
        )
        return self


def _validate_step_reward_clip_bounds(
    *,
    min_value: float | None,
    max_value: float | None,
) -> None:
    if min_value is not None and max_value is not None and min_value > max_value:
        raise ValueError("step_reward_clip_min must be <= step_reward_clip_max")


class EmulatorConfig(BaseModel):
    """Paths used to boot the libretro core, content, and optional state."""

    model_config = ConfigDict(extra="forbid")

    core_path: FilePath
    rom_path: FilePath
    runtime_dir: Path | None = None
    baseline_state_path: Path | None = None
    renderer: Literal["angrylion", "gliden64"] = "angrylion"
