# src/rl_fzerox/core/runtime_spec/schema/rewards.py
"""Reward-shaping runtime schema and validation helpers.

Reward config is shared by training, evaluation, watch playback, and env tests.
Keeping the large reward surface here makes the env rollout schema easier to
scan while preserving strict validation for every reward knob.
"""

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


class RewardCourseOverrideConfig(BaseModel):
    """Course-local reward overrides for fields in :class:`RewardConfig`.

    Every field here mirrors one reward-main weight field and defaults to
    ``None`` so omitted course-local settings inherit the base reward config.
    Keep the explicit field list in sync with ``RewardConfig``; tests pin the
    field set to catch drift when reward knobs are added.
    """

    model_config = ConfigDict(extra="forbid")

    # Gated progress reward and speed/position scaling.
    time_penalty_per_frame: float | None = None
    progress_bucket_distance: NonNegativeFloat | None = None
    progress_bucket_reward: NonNegativeFloat | None = None
    progress_reward_interval_frames: PositiveInt | None = None
    suspend_progress_while_outside_track_bounds: bool | None = None
    progress_track_distance_tolerance: NonNegativeFloat | None = None
    progress_speed_min_kph: NonNegativeFloat | None = None
    progress_speed_min_multiplier: NonNegativeFloat | None = None
    progress_speed_reference_kph: PositiveFloat | None = None
    progress_speed_max_kph: PositiveFloat | None = None
    progress_speed_max_multiplier: NonNegativeFloat | None = None
    progress_speed_curve_power: PositiveFloat | None = None
    position_progress_min_multiplier: NonNegativeFloat | None = None
    position_progress_max_multiplier: NonNegativeFloat | None = None
    # Course-boundary, lap, and KO events.
    outside_track_recovery_reward: NonNegativeFloat | None = None
    outside_track_recovery_reward_cap: NonNegativeFloat | None = None
    outside_track_recovery_airborne_grace_frames: NonNegativeInt | None = None
    outside_track_dip_penalty: float | None = Field(default=None, le=0.0)
    outside_track_dip_height_threshold: float | None = Field(default=None, le=0.0)
    lap_completion_bonus: NonNegativeFloat | None = None
    lap_position_scale: NonNegativeFloat | None = None
    ko_star_reward: NonNegativeFloat | None = None
    # Surface, energy, action-request, boost-pad, and landing shaping.
    energy_loss_epsilon: NonNegativeFloat | None = None
    energy_refill_progress_multiplier: float | None = Field(default=None, ge=1.0)
    dirt_progress_multiplier: float | None = Field(default=None, ge=0.0)
    ice_progress_multiplier: float | None = Field(default=None, ge=0.0)
    dirt_entry_penalty: float | None = Field(default=None, le=0.0)
    ice_entry_penalty: float | None = Field(default=None, le=0.0)
    energy_refill_collision_cooldown_frames: NonNegativeInt | None = None
    air_brake_request_penalty: float | None = Field(default=None, le=0.0)
    spin_request_penalty: float | None = Field(default=None, le=0.0)
    lean_request_penalty: float | None = Field(default=None, le=0.0)
    lean_activation_penalty: float | None = Field(default=None, le=0.0)
    grounded_pitch_penalty: float | None = Field(default=None, le=0.0)
    impact_frame_penalty: float | None = Field(default=None, le=0.0)
    energy_loss_penalty: float | None = Field(default=None, le=0.0)
    energy_gain_reward: NonNegativeFloat | None = None
    airborne_landing_reward: float | None = None
    airborne_landing_grace_frames: NonNegativeInt | None = None
    airborne_landing_min_peak_height: NonNegativeFloat | None = None
    manual_boost_reward: NonNegativeFloat | None = None
    manual_boost_reward_energy_shaping: bool | None = None
    manual_boost_reward_min_energy_fraction: float | None = Field(default=None, ge=0.0, lt=1.0)
    manual_boost_reward_min_energy_value: float | None = None
    manual_boost_reward_full_energy_fraction: float | None = Field(default=None, gt=0.0, le=1.0)
    manual_boost_reward_energy_curve: Literal["linear", "smoothstep"] | None = None
    boost_pad_reward_cannot_boost: NonNegativeFloat | None = None
    boost_pad_reward_can_boost: NonNegativeFloat | None = None
    boost_pad_reward_progress_window: PositiveFloat | None = None
    # Episode termination and optional final clipping.
    failure_penalty: float | None = None
    truncation_penalty: float | None = None
    step_reward_clip_min: float | None = None
    step_reward_clip_max: float | None = None

    @model_validator(mode="after")
    def _validate_reward_bounds(self) -> RewardCourseOverrideConfig:
        _validate_step_reward_clip_bounds(
            min_value=self.step_reward_clip_min,
            max_value=self.step_reward_clip_max,
        )
        _validate_progress_speed_bounds(
            min_kph=self.progress_speed_min_kph,
            reference_kph=self.progress_speed_reference_kph,
            max_kph=self.progress_speed_max_kph,
        )
        _validate_position_progress_bounds(
            min_multiplier=self.position_progress_min_multiplier,
            max_multiplier=self.position_progress_max_multiplier,
        )
        _validate_manual_boost_energy_bounds(
            min_fraction=self.manual_boost_reward_min_energy_fraction,
            full_fraction=self.manual_boost_reward_full_energy_fraction,
        )
        return self


class RewardConfig(BaseModel):
    """Reward-shaping settings for the current env."""

    model_config = ConfigDict(extra="forbid")

    name: Literal["reward_main"] = "reward_main"
    # Gated progress reward and speed/position scaling.
    time_penalty_per_frame: float = -0.005
    progress_bucket_distance: NonNegativeFloat = 1_000.0
    progress_bucket_reward: NonNegativeFloat = 1.0
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
    # Course-boundary, lap, and KO events.
    outside_track_recovery_reward: NonNegativeFloat = 0.0
    outside_track_recovery_reward_cap: NonNegativeFloat = 0.1
    outside_track_recovery_airborne_grace_frames: NonNegativeInt = 30
    outside_track_dip_penalty: float = Field(default=0.0, le=0.0)
    outside_track_dip_height_threshold: float = Field(default=0.0, le=0.0)
    lap_completion_bonus: NonNegativeFloat = 5.0
    lap_position_scale: NonNegativeFloat = 1.0
    ko_star_reward: NonNegativeFloat = 0.0
    # Surface, energy, action-request, boost-pad, and landing shaping.
    energy_loss_epsilon: NonNegativeFloat = 0.01
    energy_refill_progress_multiplier: float = Field(default=1.0, ge=1.0)
    dirt_progress_multiplier: float = Field(default=1.0, ge=0.0)
    ice_progress_multiplier: float = Field(default=1.0, ge=0.0)
    dirt_entry_penalty: float = Field(default=0.0, le=0.0)
    ice_entry_penalty: float = Field(default=0.0, le=0.0)
    energy_refill_collision_cooldown_frames: NonNegativeInt = 0
    air_brake_request_penalty: float = Field(default=0.0, le=0.0)
    spin_request_penalty: float = Field(default=0.0, le=0.0)
    lean_request_penalty: float = Field(default=0.0, le=0.0)
    lean_activation_penalty: float = Field(default=0.0, le=0.0)
    grounded_pitch_penalty: float = Field(default=0.0, le=0.0)
    impact_frame_penalty: float = Field(default=-0.02, le=0.0)
    energy_loss_penalty: float = Field(default=-0.01, le=0.0)
    energy_gain_reward: NonNegativeFloat = 0.01
    airborne_landing_reward: float = 0.0
    airborne_landing_grace_frames: NonNegativeInt = 50
    airborne_landing_min_peak_height: NonNegativeFloat = 50.0
    manual_boost_reward: NonNegativeFloat = 0.0
    manual_boost_reward_energy_shaping: bool = False
    manual_boost_reward_min_energy_fraction: float = Field(default=0.0, ge=0.0, lt=1.0)
    manual_boost_reward_min_energy_value: float = 0.0
    manual_boost_reward_full_energy_fraction: float = Field(default=1.0, gt=0.0, le=1.0)
    manual_boost_reward_energy_curve: Literal["linear", "smoothstep"] = "linear"
    boost_pad_reward_cannot_boost: NonNegativeFloat = 0.0
    boost_pad_reward_can_boost: NonNegativeFloat = 0.0
    boost_pad_reward_progress_window: PositiveFloat = 1_000.0
    # Episode termination, optional final clipping, and course-local overrides.
    failure_penalty: float = -20.0
    truncation_penalty: float = -20.0
    step_reward_clip_min: float | None = None
    step_reward_clip_max: float | None = None
    course_overrides: dict[str, RewardCourseOverrideConfig] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_reward_bounds(self) -> RewardConfig:
        _validate_step_reward_clip_bounds(
            min_value=self.step_reward_clip_min,
            max_value=self.step_reward_clip_max,
        )
        _validate_progress_speed_bounds(
            min_kph=self.progress_speed_min_kph,
            reference_kph=self.progress_speed_reference_kph,
            max_kph=self.progress_speed_max_kph,
        )
        _validate_position_progress_bounds(
            min_multiplier=self.position_progress_min_multiplier,
            max_multiplier=self.position_progress_max_multiplier,
        )
        _validate_manual_boost_energy_bounds(
            min_fraction=self.manual_boost_reward_min_energy_fraction,
            full_fraction=self.manual_boost_reward_full_energy_fraction,
        )
        return self


def _validate_step_reward_clip_bounds(
    *,
    min_value: float | None,
    max_value: float | None,
) -> None:
    if min_value is not None and max_value is not None and min_value > max_value:
        raise ValueError("step_reward_clip_min must be <= step_reward_clip_max")


def _validate_progress_speed_bounds(
    *,
    min_kph: float | None,
    reference_kph: float | None,
    max_kph: float | None,
) -> None:
    if min_kph is not None and reference_kph is not None and reference_kph <= min_kph:
        raise ValueError("progress_speed_reference_kph must be greater than min kph")
    if reference_kph is not None and max_kph is not None and max_kph <= reference_kph:
        raise ValueError("progress_speed_max_kph must be greater than reference kph")


def _validate_position_progress_bounds(
    *,
    min_multiplier: float | None,
    max_multiplier: float | None,
) -> None:
    if (
        min_multiplier is not None
        and max_multiplier is not None
        and min_multiplier > max_multiplier
    ):
        raise ValueError(
            "position_progress_min_multiplier must be <= position_progress_max_multiplier"
        )


def _validate_manual_boost_energy_bounds(
    *,
    min_fraction: float | None,
    full_fraction: float | None,
) -> None:
    if min_fraction is not None and full_fraction is not None and min_fraction >= full_fraction:
        raise ValueError(
            "manual_boost_reward_min_energy_fraction must be less than "
            "manual_boost_reward_full_energy_fraction"
        )
