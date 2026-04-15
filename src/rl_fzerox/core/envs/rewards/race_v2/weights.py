# src/rl_fzerox/core/envs/rewards/race_v2/weights.py
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RaceV2RewardWeights:
    """Weights for the race-first reward profile (`race_v2`)."""

    time_penalty_per_frame: float = -0.005
    reverse_time_penalty_scale: float = 2.0
    low_speed_time_penalty_scale: float = 2.0
    milestone_distance: float = 3_000.0
    randomize_milestone_phase_on_reset: bool = False
    milestone_bonus: float = 2.0
    milestone_speed_scale: float = 0.0
    milestone_speed_bonus_cap: float = 0.0
    bootstrap_progress_scale: float = 0.001
    bootstrap_regress_penalty_scale: float = 0.002
    progress_reward_interval_frames: int = 1
    bootstrap_position_multiplier_scale: float = 0.0
    bootstrap_lap_count: int = 1
    lap_1_completion_bonus: float = 20.0
    lap_2_completion_bonus: float = 35.0
    final_lap_completion_bonus: float = 60.0
    lap_position_scale: float = 1.0
    remaining_step_penalty_per_frame: float = 0.01
    remaining_lap_penalty: float = 50.0
    energy_loss_epsilon: float = 0.01
    energy_loss_penalty_scale: float = 0.05
    energy_loss_safe_fraction: float = 0.9
    energy_loss_danger_power: float = 2.0
    energy_gain_reward_scale: float = 0.02
    energy_gain_collision_cooldown_frames: int = 0
    energy_full_refill_bonus: float = 0.0
    energy_full_refill_cooldown_frames: int = 0
    damage_taken_frame_penalty: float = 0.0
    damage_taken_streak_ramp_penalty: float = 0.0
    damage_taken_streak_cap_frames: int = 0
    airborne_landing_reward: float = 0.0
    grounded_air_brake_penalty: float = 0.0
    drive_axis_negative_penalty_scale: float = 0.0
    boost_pad_reward: float = 0.0
    boost_pad_reward_cooldown_frames: int = 0
    manual_boost_request_reward: float = 0.0
    collision_recoil_penalty: float = -2.0
    spinning_out_penalty: float = -4.0
    terminal_failure_base_penalty: float = -120.0
    stuck_truncation_base_penalty: float = -150.0
    wrong_way_truncation_base_penalty: float = -170.0
    progress_stalled_truncation_base_penalty: float = -150.0
    timeout_truncation_base_penalty: float = -150.0
    finish_position_scale: float = 4.0
