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
    stuck_truncation_enabled: bool = True
    stuck_step_limit: PositiveInt = 240
    stuck_min_speed_kph: NonNegativeFloat = 50.0
    wrong_way_truncation_enabled: bool = True
    wrong_way_timer_limit: PositiveInt = 300
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
    defer_progress_reward_while_airborne: bool = False
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


class EmulatorConfig(BaseModel):
    """Paths used to boot the libretro core, content, and optional state."""

    model_config = ConfigDict(extra="forbid")

    core_path: FilePath
    rom_path: FilePath
    runtime_dir: Path | None = None
    baseline_state_path: Path | None = None
    renderer: Literal["angrylion", "gliden64"] = "angrylion"
