# tests/core/game/test_reward_config.py
from __future__ import annotations

from dataclasses import fields

from rl_fzerox.core.envs.rewards import (
    CANONICAL_REWARD_NAME,
    RewardMainTracker,
    RewardMainWeights,
    build_reward_tracker,
    reward_tracker_names,
)
from rl_fzerox.core.runtime_spec.schema import RewardConfig

_COURSE_EFFECT_PIT = 1
_COURSE_EFFECT_DIRT = 2
_COURSE_EFFECT_DASH = 3
_COURSE_EFFECT_ICE = 4


def test_default_reward_config_uses_canonical_profile() -> None:
    assert RewardConfig().name == CANONICAL_REWARD_NAME
    assert RewardConfig().name in reward_tracker_names()


def test_registered_reward_profiles_only_include_canonical_name() -> None:
    assert CANONICAL_REWARD_NAME == "reward_main"
    assert reward_tracker_names() == ("reward_main",)
    assert isinstance(build_reward_tracker(), RewardMainTracker)


def test_reward_main_weight_fields_match_reward_config_schema() -> None:
    weight_fields = {field.name for field in fields(RewardMainWeights)}
    schema_fields = set(RewardConfig.model_fields) - {"name", "course_overrides"}
    assert weight_fields == schema_fields


def test_build_reward_tracker_wires_all_reward_main_weight_fields() -> None:
    overrides = {
        "energy_loss_epsilon": 0.04,
        "progress_bucket_distance": 123.0,
        "progress_bucket_reward": 2.5,
        "progress_reward_interval_frames": 7,
        "suspend_progress_while_outside_track_bounds": False,
        "progress_track_distance_tolerance": 250.0,
        "progress_speed_min_kph": 100.0,
        "progress_speed_min_multiplier": 0.25,
        "progress_speed_reference_kph": 760.0,
        "progress_speed_max_kph": 1_500.0,
        "progress_speed_max_multiplier": 1.5,
        "progress_speed_curve_power": 2.0,
        "position_progress_min_multiplier": 0.8,
        "position_progress_max_multiplier": 1.25,
        "outside_track_recovery_reward": 0.125,
        "outside_track_recovery_reward_cap": 0.75,
        "outside_track_recovery_airborne_grace_frames": 11,
        "time_penalty_per_frame": -0.002,
        "lap_completion_bonus": 9.0,
        "lap_position_scale": 0.33,
        "ko_star_reward": 4.0,
        "impact_frame_penalty": -0.02,
        "energy_loss_penalty": -0.01,
        "energy_gain_reward": 0.01,
        "manual_boost_reward": 0.25,
        "manual_boost_reward_energy_shaping": True,
        "manual_boost_reward_min_energy_fraction": 0.2,
        "manual_boost_reward_min_energy_value": 0.025,
        "manual_boost_reward_full_energy_fraction": 0.8,
        "manual_boost_reward_energy_curve": "smoothstep",
        "boost_pad_reward": 10.0,
        "boost_pad_reward_progress_window": 800.0,
        "energy_refill_progress_multiplier": 3.0,
        "dirt_progress_multiplier": 0.5,
        "ice_progress_multiplier": 0.75,
        "dirt_entry_penalty": -0.5,
        "ice_entry_penalty": -0.25,
        "energy_refill_collision_cooldown_frames": 17,
        "air_brake_request_penalty": -0.005,
        "spin_request_penalty": -0.006,
        "lean_request_penalty": -0.002,
        "lean_activation_penalty": -0.01,
        "grounded_pitch_penalty": -0.004,
        "airborne_landing_reward": 5.0,
        "airborne_landing_grace_frames": 33,
        "airborne_landing_min_peak_height": 120.0,
        "failure_penalty": -30.0,
        "truncation_penalty": -15.0,
        "step_reward_clip_min": -12.0,
        "step_reward_clip_max": 18.0,
    }
    assert set(overrides) == {field.name for field in fields(RewardMainWeights)}

    tracker = build_reward_tracker(RewardConfig(**overrides))

    assert isinstance(tracker, RewardMainTracker)
    actual = {
        field.name: getattr(tracker._weights, field.name) for field in fields(RewardMainWeights)
    }
    assert actual == overrides
