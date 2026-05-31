# src/rl_fzerox/core/manager/run_spec/sections/reward.py
"""Reward section of the manager-owned run-spec model."""

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


class ManagedRewardConfig(BaseModel):
    """Reward knobs exposed by the run manager."""

    model_config = ConfigDict(extra="forbid")

    time_penalty_per_frame: float = 0.0
    progress_bucket_distance: PositiveFloat = 25.0
    progress_bucket_reward: NonNegativeFloat = 0.05
    progress_reward_interval_frames: PositiveInt = 1
    suspend_progress_while_outside_track_bounds: bool = True
    progress_track_distance_tolerance: NonNegativeFloat = 1_000.0
    progress_speed_min_kph: NonNegativeFloat = 0.0
    progress_speed_min_multiplier: NonNegativeFloat = 1.0
    progress_speed_reference_kph: PositiveFloat = 760.0
    progress_speed_max_kph: PositiveFloat = 1_500.0
    progress_speed_max_multiplier: NonNegativeFloat = 1.0
    progress_speed_curve_power: PositiveFloat = 1.0
    position_progress_min_multiplier: NonNegativeFloat = 1.0
    position_progress_max_multiplier: NonNegativeFloat = 1.0
    outside_track_recovery_reward: NonNegativeFloat = 0.0
    outside_track_recovery_reward_cap: NonNegativeFloat = 0.1
    outside_track_recovery_airborne_grace_frames: NonNegativeInt = 30
    lap_completion_bonus: NonNegativeFloat = 5.0
    lap_position_scale: NonNegativeFloat = 1.0
    ko_star_reward: NonNegativeFloat = 0.0
    energy_loss_epsilon: NonNegativeFloat = 0.01
    energy_refill_progress_multiplier: float = Field(default=3.0, ge=1.0)
    dirt_progress_multiplier: float = Field(default=1.0, ge=0.0)
    ice_progress_multiplier: float = Field(default=1.0, ge=0.0)
    dirt_entry_penalty: float = Field(default=0.0, le=0.0)
    ice_entry_penalty: float = Field(default=0.0, le=0.0)
    energy_refill_collision_cooldown_frames: NonNegativeInt = 120
    air_brake_request_penalty: float = Field(default=0.0, le=0.0)
    manual_boost_reward: NonNegativeFloat = 0.01
    manual_boost_reward_energy_shaping: bool = False
    manual_boost_reward_min_energy_multiplier: float = Field(default=0.0, ge=0.0, le=1.0)
    manual_boost_reward_full_energy_fraction: float = Field(default=1.0, gt=0.0, le=1.0)
    manual_boost_reward_energy_curve: Literal["linear", "smoothstep"] = "linear"
    boost_pad_reward: NonNegativeFloat = 10.0
    boost_pad_reward_progress_window: PositiveFloat = 800.0
    lean_request_penalty: float = Field(default=-0.003, le=0.0)
    lean_activation_penalty: float = Field(default=0.0, le=0.0)
    grounded_pitch_penalty: float = Field(default=0.0, le=0.0)
    impact_frame_penalty: float = Field(default=-0.02, le=0.0)
    energy_loss_penalty: float = Field(default=-0.01, le=0.0)
    energy_gain_reward: NonNegativeFloat = 0.01
    airborne_landing_reward: float = 1.0
    airborne_landing_grace_frames: NonNegativeInt = 50
    airborne_landing_min_peak_height: NonNegativeFloat = 50.0
    failure_penalty: float = -30.0
    truncation_penalty: float = -30.0
    step_reward_clip_min: float | None = -100.0
    step_reward_clip_max: float | None = 100.0

    @model_validator(mode="after")
    def _validate_reward_bounds(self) -> ManagedRewardConfig:
        if (
            self.step_reward_clip_min is not None
            and self.step_reward_clip_max is not None
            and self.step_reward_clip_min > self.step_reward_clip_max
        ):
            raise ValueError("step_reward_clip_min must be <= step_reward_clip_max")
        if self.progress_speed_reference_kph <= self.progress_speed_min_kph:
            raise ValueError("progress_speed_reference_kph must be greater than min kph")
        if self.progress_speed_max_kph <= self.progress_speed_reference_kph:
            raise ValueError("progress_speed_max_kph must be greater than reference kph")
        if self.position_progress_min_multiplier > self.position_progress_max_multiplier:
            raise ValueError(
                "position_progress_min_multiplier must be <= position_progress_max_multiplier"
            )
        return self
