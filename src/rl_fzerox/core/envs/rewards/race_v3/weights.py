# src/rl_fzerox/core/envs/rewards/race_v3/weights.py
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RaceV3RewardWeights:
    """Weights for the spline-bucket frontier coverage profile (`race_v3`)."""

    energy_loss_epsilon: float = 0.01
    progress_bucket_distance: float = 1_000.0
    progress_bucket_reward: float = 1.0
    progress_reward_interval_frames: int = 1
    defer_progress_reward_while_airborne: bool = False
    airborne_progress_bucket_distance: float | None = None
    airborne_progress_requires_nonascending: bool = False
    airborne_progress_height_epsilon: float = 0.0
    time_penalty_per_frame: float = 0.0
    reverse_time_penalty_scale: float = 1.0
    low_speed_time_penalty_scale: float = 1.0
    lap_completion_bonus: float = 5.0
    lap_position_scale: float = 0.25
    damage_taken_frame_penalty: float = -0.02
    damage_taken_streak_ramp_penalty: float = -0.001
    damage_taken_streak_cap_frames: int = 120
    manual_boost_reward: float = 0.0
    boost_pad_reward: float = 0.0
    boost_pad_reward_progress_window: float = 1_000.0
    energy_refill_progress_multiplier: float = 1.0
    dirt_progress_multiplier: float = 1.0
    ice_progress_multiplier: float = 1.0
    dirt_entry_penalty: float = 0.0
    ice_entry_penalty: float = 0.0
    energy_refill_collision_cooldown_frames: int = 120
    energy_full_refill_lap_bonus: float = 0.0
    energy_full_refill_min_gain_fraction: float = 0.0
    gas_underuse_penalty: float = 0.0
    gas_underuse_threshold: float = 0.5
    steer_oscillation_penalty: float = 0.0
    steer_oscillation_deadzone: float = 0.0
    steer_oscillation_cap: float = 2.0
    steer_oscillation_power: float = 2.0
    lean_request_penalty: float = 0.0
    lean_low_speed_penalty: float = 0.0
    lean_low_speed_penalty_max_speed_kph: float = 800.0
    airborne_landing_reward: float = 0.0
    collision_recoil_penalty: float = -0.25
    failure_penalty: float = -20.0
    truncation_penalty: float = -20.0
