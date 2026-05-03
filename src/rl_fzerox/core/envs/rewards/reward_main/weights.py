# src/rl_fzerox/core/envs/rewards/reward_main/weights.py
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RewardMainWeights:
    """Weights for the current canonical spline-bucket reward profile."""

    energy_loss_epsilon: float = 0.01
    progress_bucket_distance: float = 1_000.0
    progress_bucket_reward: float = 1.0
    progress_reward_interval_frames: int = 1
    airborne_progress_bucket_distance: float | None = None
    outside_bounds_reentry_progress_distance_cap: float | None = None
    airborne_offtrack_penalty_scale: float = 0.0
    airborne_offtrack_recovery_reward_scale: float = 0.0
    airborne_offtrack_recovery_requires_descending: bool = False
    airborne_offtrack_recovery_descend_epsilon: float = 1.0
    time_penalty_per_frame: float = 0.0
    reverse_time_penalty_scale: float = 2.0
    slow_speed_time_penalty_scale: float = 3.0
    slow_speed_time_penalty_start_kph: float = 760.0
    slow_speed_time_penalty_power: float = 1.0
    lap_completion_bonus: float = 5.0
    lap_position_scale: float = 1.0
    damage_taken_frame_penalty: float = 0.0
    damage_taken_streak_ramp_penalty: float = 0.0
    damage_taken_streak_cap_frames: int = 0
    manual_boost_reward: float = 0.01
    boost_pad_reward: float = 10.0
    boost_pad_reward_progress_window: float = 800.0
    energy_refill_progress_multiplier: float = 3.0
    dirt_progress_multiplier: float = 1.0
    ice_progress_multiplier: float = 1.0
    dirt_entry_penalty: float = 0.0
    ice_entry_penalty: float = 0.0
    energy_refill_collision_cooldown_frames: int = 120
    gas_underuse_penalty: float = 0.0
    gas_underuse_threshold: float = 0.5
    steer_oscillation_penalty: float = 0.0
    steer_oscillation_deadzone: float = 0.0
    steer_oscillation_cap: float = 2.0
    steer_oscillation_power: float = 2.0
    air_brake_request_penalty: float = 0.0
    lean_request_penalty: float = -0.003
    airborne_pitch_up_penalty: float = -0.2
    lean_low_speed_penalty: float = -0.005
    lean_low_speed_penalty_max_speed_kph: float = 750.0
    airborne_landing_reward: float = 0.0
    collision_recoil_penalty: float = -4.0
    failure_penalty: float = -30.0
    truncation_penalty: float = -30.0
    step_reward_clip_min: float | None = -100.0
    step_reward_clip_max: float | None = 100.0
