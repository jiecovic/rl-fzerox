# src/rl_fzerox/core/envs/rewards/reward_main/weights.py
"""Weight dataclass for the canonical `reward_main` profile.

The fields here are the config-facing knobs used by progress, events, energy,
control, and recovery terms. Keep defaults conservative because run-manager
configs often inherit them implicitly.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RewardMainWeights:
    """Weights for the current canonical spline-bucket reward profile."""

    energy_loss_epsilon: float = 0.01
    progress_bucket_distance: float = 25.0
    progress_bucket_reward: float = 0.05
    progress_reward_interval_frames: int = 1
    suspend_progress_while_outside_track_bounds: bool = True
    progress_track_distance_tolerance: float = 1_000.0
    progress_speed_min_kph: float = 0.0
    progress_speed_min_multiplier: float = 1.0
    progress_speed_reference_kph: float = 760.0
    progress_speed_max_kph: float = 1_500.0
    progress_speed_max_multiplier: float = 1.0
    progress_speed_curve_power: float = 1.0
    position_progress_min_multiplier: float = 1.0
    position_progress_max_multiplier: float = 1.0
    outside_track_recovery_reward: float = 0.0
    outside_track_recovery_reward_cap: float = 0.1
    outside_track_recovery_airborne_grace_frames: int = 30
    outside_track_dip_penalty: float = 0.0
    outside_track_dip_height_threshold: float = 0.0
    time_penalty_per_frame: float = 0.0
    lap_completion_bonus: float = 5.0
    lap_position_scale: float = 1.0
    ko_star_reward: float = 0.0
    impact_frame_penalty: float = -0.02
    energy_loss_penalty: float = -0.01
    energy_gain_reward: float = 0.01
    manual_boost_reward: float = 0.01
    manual_boost_reward_energy_shaping: bool = False
    manual_boost_reward_min_energy_fraction: float = 0.0
    manual_boost_reward_min_energy_value: float = 0.0
    manual_boost_reward_full_energy_fraction: float = 1.0
    manual_boost_reward_energy_curve: str = "linear"
    boost_pad_reward_cannot_boost: float = 10.0
    boost_pad_reward_can_boost: float = 10.0
    boost_pad_reward_progress_window: float = 800.0
    energy_refill_progress_multiplier: float = 3.0
    dirt_progress_multiplier: float = 1.0
    ice_progress_multiplier: float = 1.0
    dirt_entry_penalty: float = 0.0
    ice_entry_penalty: float = 0.0
    energy_refill_collision_cooldown_frames: int = 120
    air_brake_request_penalty: float = 0.0
    spin_request_penalty: float = 0.0
    lean_request_penalty: float = -0.003
    lean_activation_penalty: float = 0.0
    grounded_pitch_penalty: float = 0.0
    airborne_landing_reward: float = 1.0
    airborne_landing_grace_frames: int = 50
    airborne_landing_min_peak_height: float = 50.0
    failure_penalty: float = -30.0
    truncation_penalty: float = -30.0
    step_reward_clip_min: float | None = -100.0
    step_reward_clip_max: float | None = 100.0

    def __post_init__(self) -> None:
        if self.progress_speed_reference_kph <= self.progress_speed_min_kph:
            raise ValueError("progress_speed_reference_kph must be greater than min kph")
        if self.progress_speed_max_kph <= self.progress_speed_reference_kph:
            raise ValueError("progress_speed_max_kph must be greater than reference kph")
        if self.position_progress_min_multiplier > self.position_progress_max_multiplier:
            raise ValueError(
                "position_progress_min_multiplier must be <= position_progress_max_multiplier"
            )
        if not 0.0 <= self.manual_boost_reward_min_energy_fraction < 1.0:
            raise ValueError("manual_boost_reward_min_energy_fraction must be in [0, 1)")
        if not 0.0 < self.manual_boost_reward_full_energy_fraction <= 1.0:
            raise ValueError("manual_boost_reward_full_energy_fraction must be in (0, 1]")
        if self.manual_boost_reward_min_energy_fraction >= (
            self.manual_boost_reward_full_energy_fraction
        ):
            raise ValueError(
                "manual_boost_reward_min_energy_fraction must be less than "
                "manual_boost_reward_full_energy_fraction"
            )
        if self.manual_boost_reward_energy_curve not in {"linear", "smoothstep"}:
            raise ValueError("manual_boost_reward_energy_curve must be linear or smoothstep")
