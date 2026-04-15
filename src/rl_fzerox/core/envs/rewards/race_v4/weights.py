# src/rl_fzerox/core/envs/rewards/race_v4/weights.py
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RaceV4RewardWeights:
    """Weights for milestone reward with explicit frame-level time pressure."""

    energy_loss_epsilon: float = 0.01
    time_penalty_per_frame: float = -0.0003
    reverse_time_penalty_scale: float = 2.0
    low_speed_time_penalty_scale: float = 2.0
    milestone_distance: float = 5_000.0
    randomize_milestone_phase_on_reset: bool = True
    milestone_bonus: float = 1.0
    lap_1_completion_bonus: float = 1.0
    lap_2_completion_bonus: float = 1.0
    final_lap_completion_bonus: float = 3.0
    lap_position_scale: float = 0.0
    remaining_step_penalty_per_frame: float = 0.0
    remaining_lap_penalty: float = 10.0
    damage_taken_frame_penalty: float = -0.02
    damage_taken_streak_ramp_penalty: float = -0.001
    damage_taken_streak_cap_frames: int = 120
    terminal_failure_base_penalty: float = -20.0
    stuck_truncation_base_penalty: float = -20.0
    wrong_way_truncation_base_penalty: float = -20.0
    progress_stalled_truncation_base_penalty: float = -20.0
    timeout_truncation_base_penalty: float = -20.0
    finish_position_scale: float = 0.0
