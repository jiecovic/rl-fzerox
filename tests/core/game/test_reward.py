# tests/core/game/test_reward.py
from __future__ import annotations

from dataclasses import fields

import pytest

from fzerox_emulator import FZeroXTelemetry, StepStatus, StepSummary
from rl_fzerox.core.envs.rewards import (
    CANONICAL_REWARD_NAME,
    RewardActionContext,
    RewardMainTracker,
    RewardMainWeights,
    build_reward_tracker,
    reward_tracker_names,
)
from rl_fzerox.core.runtime_spec.schema import RewardConfig, RewardCourseOverrideConfig
from tests.support.native_objects import encode_state_flags, make_step_summary, make_telemetry

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


def test_reward_main_rewards_new_gp_ko_stars_once() -> None:
    tracker = build_reward_tracker(
        RewardConfig(
            ko_star_reward=2.5,
            progress_bucket_reward=0.0,
            time_penalty_per_frame=0.0,
            impact_frame_penalty=0.0,
        )
    )
    tracker.reset(_telemetry(ko_star_count=1))

    gained_two = tracker.step_summary(
        _summary(max_race_distance=0.0),
        _status(step_count=1),
        _telemetry(ko_star_count=3),
    )
    repeated_count = tracker.step_summary(
        _summary(max_race_distance=0.0),
        _status(step_count=2),
        _telemetry(ko_star_count=3),
    )

    assert gained_two.reward == 5.0
    assert gained_two.breakdown == {"ko_star": 5.0}
    assert repeated_count.reward == 0.0


def test_reward_main_ignores_ko_star_reward_outside_gp_race() -> None:
    tracker = build_reward_tracker(
        RewardConfig(
            ko_star_reward=2.5,
            progress_bucket_reward=0.0,
            time_penalty_per_frame=0.0,
            impact_frame_penalty=0.0,
        )
    )
    tracker.reset(_telemetry(game_mode_name="practice", ko_star_count=0))

    step = tracker.step_summary(
        _summary(max_race_distance=0.0),
        _status(step_count=1),
        _telemetry(game_mode_name="practice", ko_star_count=1),
    )

    assert step.reward == 0.0
    assert step.breakdown == {}


def test_reward_main_rewards_each_frontier_bucket_once() -> None:
    tracker = build_reward_tracker(
        RewardConfig(
            progress_bucket_distance=1_000.0,
            progress_bucket_reward=2.0,
            time_penalty_per_frame=0.0,
            impact_frame_penalty=0.0,
        )
    )
    tracker.reset(_telemetry(race_distance=0.0))

    before_first_bucket = tracker.step_summary(
        _summary(max_race_distance=999.0),
        _status(step_count=1),
        _telemetry(race_distance=999.0),
    )
    first_step = tracker.step_summary(
        _summary(max_race_distance=2_500.0),
        _status(step_count=2),
        _telemetry(race_distance=2_500.0),
    )
    repeated_step = tracker.step_summary(
        _summary(max_race_distance=2_500.0),
        _status(step_count=3),
        _telemetry(race_distance=500.0),
    )

    assert before_first_bucket.reward == 0.0
    assert first_step.reward == 4.0
    assert first_step.breakdown == {"frontier_progress": 4.0}
    assert repeated_step.reward == 0.0
    info = tracker.info(_telemetry(race_distance=2_500.0))
    assert info["frontier_progress_bucket_index"] == 2
    assert info["frontier_progress_distance"] == 2_000.0


def test_reward_main_can_delay_frontier_rewards_by_interval() -> None:
    tracker = build_reward_tracker(
        RewardConfig(
            progress_bucket_distance=100.0,
            progress_bucket_reward=1.0,
            progress_reward_interval_frames=3,
            time_penalty_per_frame=0.0,
            impact_frame_penalty=0.0,
        )
    )
    tracker.reset(_telemetry(race_distance=0.0))

    held_step = tracker.step_summary(
        _summary(max_race_distance=100.0, frames_run=1),
        _status(step_count=1),
        _telemetry(race_distance=100.0),
    )
    flushed_step = tracker.step_summary(
        _summary(max_race_distance=300.0, frames_run=2),
        _status(step_count=3),
        _telemetry(race_distance=300.0),
    )

    assert held_step.reward == 0.0
    assert flushed_step.breakdown == {"frontier_progress": 3.0}


def test_reward_main_can_suspend_frontier_progress_while_outside_track_bounds() -> None:
    tracker = build_reward_tracker(
        RewardConfig(
            progress_bucket_distance=100.0,
            progress_bucket_reward=1.0,
            suspend_progress_while_outside_track_bounds=True,
            time_penalty_per_frame=0.0,
            impact_frame_penalty=0.0,
        )
    )
    tracker.reset(_telemetry(race_distance=0.0))

    airborne = tracker.step_summary(
        _summary(max_race_distance=350.0),
        _status(step_count=1),
        _telemetry(race_distance=350.0, state_labels=("active", "airborne")),
    )
    outside_bounds = tracker.step_summary(
        _summary(max_race_distance=450.0),
        _status(step_count=2),
        _telemetry(
            race_distance=450.0,
            state_labels=("active", "airborne"),
            signed_lateral_offset=150.0,
            current_radius_left=100.0,
            future_local_nearest_segment_index=12,
            future_local_nearest_segment_distance=1_500.0,
        ),
    )
    landing = tracker.step_summary(
        _summary(max_race_distance=500.0),
        _status(step_count=3),
        _telemetry(race_distance=500.0),
    )

    assert airborne.breakdown == {"frontier_progress": 3.0}
    assert outside_bounds.reward == 0.0
    assert outside_bounds.breakdown == {}
    assert landing.breakdown == {"frontier_progress": 1.0}


def test_reward_main_keeps_frontier_progress_near_track_bounds() -> None:
    tracker = build_reward_tracker(
        RewardConfig(
            progress_bucket_distance=100.0,
            progress_bucket_reward=1.0,
            progress_track_distance_tolerance=200.0,
            time_penalty_per_frame=0.0,
            impact_frame_penalty=0.0,
        )
    )
    tracker.reset(_telemetry(race_distance=0.0, current_radius_left=100.0))

    near_bounds = tracker.step_summary(
        _summary(max_race_distance=300.0),
        _status(step_count=1),
        _telemetry(
            race_distance=300.0,
            signed_lateral_offset=150.0,
            current_radius_left=100.0,
            future_local_nearest_segment_index=12,
            future_local_nearest_segment_distance=150.0,
        ),
    )

    assert near_bounds.reward == 3.0
    assert near_bounds.breakdown == {"frontier_progress": 3.0}


def test_reward_main_skips_far_off_track_progress_without_reentry_payout() -> None:
    tracker = build_reward_tracker(
        RewardConfig(
            progress_bucket_distance=100.0,
            progress_bucket_reward=1.0,
            progress_track_distance_tolerance=200.0,
            time_penalty_per_frame=0.0,
            impact_frame_penalty=0.0,
        )
    )
    tracker.reset(_telemetry(race_distance=0.0, current_radius_left=100.0))

    tracker.step_summary(
        _summary(max_race_distance=1_800.0),
        _status(step_count=1),
        _telemetry(
            race_distance=1_800.0,
            signed_lateral_offset=150.0,
            current_radius_left=100.0,
            future_local_nearest_segment_index=12,
            future_local_nearest_segment_distance=1_000.0,
        ),
    )
    reentry = tracker.step_summary(
        _summary(max_race_distance=1_800.0),
        _status(step_count=2),
        _telemetry(
            race_distance=1_300.0,
            signed_lateral_offset=80.0,
            current_radius_left=100.0,
        ),
    )

    assert reentry.reward == 0.0
    assert reentry.breakdown == {}
    info = tracker.info(_telemetry(race_distance=1_350.0, current_radius_left=100.0))
    assert info["frontier_progress_distance"] == 1_800.0


def test_reward_main_clips_final_step_reward_after_breakdown_terms() -> None:
    tracker = build_reward_tracker(
        RewardConfig(
            progress_bucket_distance=100.0,
            progress_bucket_reward=1.0,
            step_reward_clip_min=-3.0,
            step_reward_clip_max=2.0,
            time_penalty_per_frame=0.0,
            impact_frame_penalty=0.0,
        )
    )
    tracker.reset(_telemetry(race_distance=0.0))

    positive = tracker.step_summary(
        _summary(max_race_distance=1_000.0),
        _status(step_count=1),
        _telemetry(race_distance=1_000.0),
    )
    negative = tracker.step_summary(
        _summary(max_race_distance=1_000.0),
        _status(step_count=2, termination_reason="crashed"),
        _telemetry(race_distance=1_000.0),
    )

    assert positive.raw_reward == 10.0
    assert positive.reward == 2.0
    assert positive.breakdown == {"frontier_progress": 10.0, "step_reward_clip": -8.0}
    assert negative.raw_reward == -20.0
    assert negative.reward == -3.0
    assert negative.breakdown == {"crashed": -20.0, "step_reward_clip": 17.0}


def test_reward_main_shapes_outside_track_recovery_by_direction() -> None:
    tracker = build_reward_tracker(
        RewardConfig(
            progress_bucket_reward=0.0,
            outside_track_recovery_reward=0.0001,
            outside_track_recovery_reward_cap=1.0,
            outside_track_recovery_airborne_grace_frames=0,
            time_penalty_per_frame=0.0,
            impact_frame_penalty=0.0,
        )
    )
    tracker.reset(_telemetry(race_distance=0.0, current_radius_left=100.0))

    first_outside = tracker.step_summary(
        _summary(max_race_distance=0.0, airborne_frames=5),
        _status(step_count=1),
        _telemetry(
            race_distance=0.0,
            state_labels=("active", "airborne"),
            signed_lateral_offset=150.0,
            lateral_distance=150.0,
            current_radius_left=100.0,
            future_local_nearest_segment_index=12,
            future_local_nearest_segment_distance=150.0,
        ),
    )
    recovering = tracker.step_summary(
        _summary(max_race_distance=0.0, airborne_frames=5),
        _status(step_count=2),
        _telemetry(
            race_distance=0.0,
            state_labels=("active", "airborne"),
            signed_lateral_offset=120.0,
            lateral_distance=120.0,
            current_radius_left=100.0,
            future_local_nearest_segment_index=12,
            future_local_nearest_segment_distance=120.0,
        ),
    )
    worsening = tracker.step_summary(
        _summary(max_race_distance=0.0, airborne_frames=5),
        _status(step_count=3),
        _telemetry(
            race_distance=0.0,
            state_labels=("active", "airborne"),
            signed_lateral_offset=140.0,
            lateral_distance=140.0,
            current_radius_left=100.0,
            future_local_nearest_segment_index=12,
            future_local_nearest_segment_distance=140.0,
        ),
    )
    back_inside = tracker.step_summary(
        _summary(max_race_distance=0.0, airborne_frames=5),
        _status(step_count=4),
        _telemetry(
            race_distance=0.0,
            signed_lateral_offset=80.0,
            lateral_distance=80.0,
            current_radius_left=100.0,
        ),
    )
    outside_again = tracker.step_summary(
        _summary(max_race_distance=0.0, airborne_frames=5),
        _status(step_count=5),
        _telemetry(
            race_distance=0.0,
            state_labels=("active", "airborne"),
            signed_lateral_offset=120.0,
            lateral_distance=120.0,
            current_radius_left=100.0,
            future_local_nearest_segment_index=12,
            future_local_nearest_segment_distance=120.0,
        ),
    )

    assert first_outside.reward == pytest.approx(-0.015)
    assert first_outside.breakdown == {"outside_track_recovery": pytest.approx(-0.015)}
    assert recovering.reward == pytest.approx(0.003)
    assert recovering.breakdown == {"outside_track_recovery": pytest.approx(0.003)}
    assert worsening.reward == pytest.approx(-0.002)
    assert worsening.breakdown == {"outside_track_recovery": pytest.approx(-0.002)}
    assert back_inside.reward == pytest.approx(0.014)
    assert back_inside.breakdown == {"outside_track_recovery": pytest.approx(0.014)}
    assert first_outside.reward + recovering.reward + worsening.reward + back_inside.reward == (
        pytest.approx(0.0)
    )
    assert outside_again.reward == pytest.approx(-0.012)
    assert outside_again.breakdown == {"outside_track_recovery": pytest.approx(-0.012)}


def test_reward_main_uses_future_segment_distance_for_outside_track_recovery() -> None:
    tracker = build_reward_tracker(
        RewardConfig(
            progress_bucket_reward=0.0,
            outside_track_recovery_reward=0.0001,
            outside_track_recovery_reward_cap=1.0,
            outside_track_recovery_airborne_grace_frames=0,
            time_penalty_per_frame=0.0,
            impact_frame_penalty=0.0,
        )
    )
    tracker.reset(
        _telemetry(
            race_distance=0.0,
            current_radius_left=100.0,
            signed_lateral_offset=150.0,
            lateral_distance=150.0,
            future_local_nearest_segment_index=12,
            future_local_nearest_segment_distance=150.0,
        )
    )

    tracker.step_summary(
        _summary(max_race_distance=0.0, airborne_frames=5),
        _status(step_count=1),
        _telemetry(
            race_distance=0.0,
            current_radius_left=100.0,
            state_labels=("active", "airborne"),
            signed_lateral_offset=150.0,
            lateral_distance=150.0,
            future_local_nearest_segment_index=12,
            future_local_nearest_segment_distance=150.0,
        ),
    )
    worsening_by_signed_offset = tracker.step_summary(
        _summary(max_race_distance=0.0, airborne_frames=5),
        _status(step_count=2),
        _telemetry(
            race_distance=0.0,
            current_radius_left=100.0,
            state_labels=("active", "airborne"),
            signed_lateral_offset=180.0,
            lateral_distance=120.0,
            future_local_nearest_segment_index=12,
            future_local_nearest_segment_distance=180.0,
        ),
    )

    assert worsening_by_signed_offset.reward == pytest.approx(-0.003)
    assert worsening_by_signed_offset.breakdown == {"outside_track_recovery": pytest.approx(-0.003)}


def test_reward_main_does_not_scale_recovery_by_active_side_radius() -> None:
    tracker = build_reward_tracker(
        RewardConfig(
            progress_bucket_reward=0.0,
            outside_track_recovery_reward=0.01,
            outside_track_recovery_reward_cap=10.0,
            outside_track_recovery_airborne_grace_frames=0,
            time_penalty_per_frame=0.0,
            impact_frame_penalty=0.0,
        )
    )
    tracker.reset(_telemetry(race_distance=0.0, current_radius_left=200.0))

    tracker.step_summary(
        _summary(max_race_distance=0.0, airborne_frames=5),
        _status(step_count=1),
        _telemetry(
            race_distance=0.0,
            state_labels=("active", "airborne"),
            signed_lateral_offset=-150.0,
            current_radius_left=200.0,
            current_radius_right=50.0,
            future_local_nearest_segment_index=12,
            future_local_nearest_segment_distance=150.0,
        ),
    )
    recovered_on_right_side = tracker.step_summary(
        _summary(max_race_distance=0.0, airborne_frames=5),
        _status(step_count=2),
        _telemetry(
            race_distance=0.0,
            state_labels=("active", "airborne"),
            signed_lateral_offset=-75.0,
            current_radius_left=200.0,
            current_radius_right=50.0,
            future_local_nearest_segment_index=12,
            future_local_nearest_segment_distance=75.0,
        ),
    )

    assert recovered_on_right_side.reward == pytest.approx(0.75)
    assert recovered_on_right_side.breakdown == {"outside_track_recovery": pytest.approx(0.75)}


def test_reward_main_caps_outside_track_recovery_reward_after_weight() -> None:
    tracker = build_reward_tracker(
        RewardConfig(
            progress_bucket_reward=0.0,
            outside_track_recovery_reward=1.0,
            outside_track_recovery_reward_cap=0.1,
            outside_track_recovery_airborne_grace_frames=0,
            time_penalty_per_frame=0.0,
            impact_frame_penalty=0.0,
        )
    )
    tracker.reset(_telemetry(race_distance=0.0, current_radius_left=100.0))

    tracker.step_summary(
        _summary(max_race_distance=0.0, airborne_frames=5),
        _status(step_count=1),
        _telemetry(
            race_distance=0.0,
            state_labels=("active", "airborne"),
            signed_lateral_offset=150.0,
            current_radius_left=100.0,
            future_local_nearest_segment_index=12,
            future_local_nearest_segment_distance=150.0,
        ),
    )
    large_recovery = tracker.step_summary(
        _summary(max_race_distance=0.0, airborne_frames=5),
        _status(step_count=2),
        _telemetry(
            race_distance=0.0,
            state_labels=("active", "airborne"),
            signed_lateral_offset=10.0,
            current_radius_left=100.0,
        ),
    )
    tracker.step_summary(
        _summary(max_race_distance=0.0, airborne_frames=5),
        _status(step_count=3),
        _telemetry(
            race_distance=0.0,
            state_labels=("active", "airborne"),
            signed_lateral_offset=150.0,
            current_radius_left=100.0,
            future_local_nearest_segment_index=12,
            future_local_nearest_segment_distance=150.0,
        ),
    )
    large_worsening = tracker.step_summary(
        _summary(max_race_distance=0.0, airborne_frames=5),
        _status(step_count=4),
        _telemetry(
            race_distance=0.0,
            state_labels=("active", "airborne"),
            signed_lateral_offset=260.0,
            current_radius_left=100.0,
            future_local_nearest_segment_index=12,
            future_local_nearest_segment_distance=260.0,
        ),
    )

    assert large_recovery.reward == pytest.approx(0.1)
    assert large_recovery.breakdown == {"outside_track_recovery": pytest.approx(0.1)}
    assert large_worsening.reward == pytest.approx(-0.1)
    assert large_worsening.breakdown == {"outside_track_recovery": pytest.approx(-0.1)}


def test_reward_main_keeps_short_airborne_outside_excursions_ungated() -> None:
    tracker = build_reward_tracker(
        RewardConfig(
            progress_bucket_reward=0.0,
            outside_track_recovery_reward=0.0001,
            outside_track_recovery_reward_cap=1.0,
            outside_track_recovery_airborne_grace_frames=30,
            time_penalty_per_frame=0.0,
            impact_frame_penalty=0.0,
        )
    )
    tracker.reset(_telemetry(race_distance=0.0, current_radius_left=100.0))

    first_outside = tracker.step_summary(
        _summary(max_race_distance=0.0, airborne_frames=10),
        _status(step_count=1),
        _telemetry(
            race_distance=0.0,
            state_labels=("active", "airborne"),
            signed_lateral_offset=150.0,
            lateral_distance=150.0,
            current_radius_left=100.0,
            future_local_nearest_segment_index=12,
            future_local_nearest_segment_distance=150.0,
        ),
    )
    recovering_too_early = tracker.step_summary(
        _summary(max_race_distance=0.0, airborne_frames=10),
        _status(step_count=2),
        _telemetry(
            race_distance=0.0,
            state_labels=("active", "airborne"),
            signed_lateral_offset=120.0,
            lateral_distance=120.0,
            current_radius_left=100.0,
            future_local_nearest_segment_index=12,
            future_local_nearest_segment_distance=120.0,
        ),
    )
    landed_still_outside = tracker.step_summary(
        _summary(max_race_distance=0.0),
        _status(step_count=3),
        _telemetry(
            race_distance=0.0,
            signed_lateral_offset=100.0,
            lateral_distance=100.0,
            current_radius_left=100.0,
        ),
    )

    assert first_outside.breakdown == {}
    assert recovering_too_early.reward == 0.0
    assert recovering_too_early.breakdown == {}
    assert landed_still_outside.reward == 0.0
    assert landed_still_outside.breakdown == {}


def test_reward_main_arms_airborne_outside_recovery_after_grace() -> None:
    tracker = build_reward_tracker(
        RewardConfig(
            progress_bucket_reward=0.0,
            outside_track_recovery_reward=0.0001,
            outside_track_recovery_reward_cap=1.0,
            outside_track_recovery_airborne_grace_frames=30,
            time_penalty_per_frame=0.0,
            impact_frame_penalty=0.0,
        )
    )
    tracker.reset(_telemetry(race_distance=0.0, current_radius_left=100.0))

    tracker.step_summary(
        _summary(max_race_distance=0.0, airborne_frames=10),
        _status(step_count=1),
        _telemetry(
            race_distance=0.0,
            state_labels=("active", "airborne"),
            signed_lateral_offset=150.0,
            lateral_distance=150.0,
            current_radius_left=100.0,
            future_local_nearest_segment_index=12,
            future_local_nearest_segment_distance=150.0,
        ),
    )
    tracker.step_summary(
        _summary(max_race_distance=0.0, airborne_frames=10),
        _status(step_count=2),
        _telemetry(
            race_distance=0.0,
            state_labels=("active", "airborne"),
            signed_lateral_offset=120.0,
            lateral_distance=120.0,
            current_radius_left=100.0,
            future_local_nearest_segment_index=12,
            future_local_nearest_segment_distance=120.0,
        ),
    )
    recovering_after_grace = tracker.step_summary(
        _summary(max_race_distance=0.0, airborne_frames=10),
        _status(step_count=3),
        _telemetry(
            race_distance=0.0,
            state_labels=("active", "airborne"),
            signed_lateral_offset=111.0,
            lateral_distance=90.0,
            current_radius_left=100.0,
            future_local_nearest_segment_index=12,
            future_local_nearest_segment_distance=110.0,
        ),
    )
    landing_inside = tracker.step_summary(
        _summary(max_race_distance=0.0, airborne_frames=5),
        _status(step_count=4),
        _telemetry(
            race_distance=0.0,
            signed_lateral_offset=80.0,
            lateral_distance=80.0,
            current_radius_left=100.0,
        ),
    )

    assert recovering_after_grace.reward == pytest.approx(-0.011)
    assert recovering_after_grace.breakdown == {"outside_track_recovery": pytest.approx(-0.011)}
    assert landing_inside.reward == pytest.approx(0.011)
    assert landing_inside.breakdown == {"outside_track_recovery": pytest.approx(0.011)}


def test_reward_main_scales_progress_by_frontier_speed() -> None:
    tracker = build_reward_tracker(
        RewardConfig(
            progress_bucket_distance=100.0,
            progress_bucket_reward=1.0,
            progress_speed_min_multiplier=0.0,
            progress_speed_reference_kph=760.0,
            progress_speed_max_kph=1_500.0,
            progress_speed_max_multiplier=2.0,
            progress_speed_curve_power=1.0,
            time_penalty_per_frame=0.0,
            impact_frame_penalty=0.0,
        )
    )
    tracker.reset(_telemetry(race_distance=0.0, speed_kph=760.0))

    half_reference_speed = tracker.step_summary(
        _summary(max_race_distance=100.0, max_race_distance_speed_kph=380.0),
        _status(step_count=1),
        _telemetry(race_distance=100.0, speed_kph=380.0),
    )
    half_top_speed_bonus = tracker.step_summary(
        _summary(max_race_distance=200.0, max_race_distance_speed_kph=1_130.0),
        _status(step_count=2),
        _telemetry(race_distance=200.0, speed_kph=1_130.0),
    )

    assert half_reference_speed.reward == pytest.approx(0.5)
    assert half_reference_speed.breakdown == {
        "frontier_progress": 1.0,
        "speed_progress": pytest.approx(-0.5),
    }
    assert half_top_speed_bonus.reward == pytest.approx(1.5)
    assert half_top_speed_bonus.breakdown == {
        "frontier_progress": 1.0,
        "speed_progress": pytest.approx(0.5),
    }


def test_reward_main_speed_curve_power_eases_out_above_reference_speed() -> None:
    tracker = build_reward_tracker(
        RewardConfig(
            progress_bucket_distance=100.0,
            progress_bucket_reward=1.0,
            progress_speed_min_multiplier=0.0,
            progress_speed_reference_kph=1_000.0,
            progress_speed_max_kph=2_000.0,
            progress_speed_max_multiplier=2.0,
            progress_speed_curve_power=2.0,
            time_penalty_per_frame=0.0,
            impact_frame_penalty=0.0,
        )
    )
    tracker.reset(_telemetry(race_distance=0.0, speed_kph=1_000.0))

    half_reference_speed = tracker.step_summary(
        _summary(max_race_distance=100.0, max_race_distance_speed_kph=500.0),
        _status(step_count=1),
        _telemetry(race_distance=100.0, speed_kph=500.0),
    )
    half_top_speed_bonus = tracker.step_summary(
        _summary(max_race_distance=200.0, max_race_distance_speed_kph=1_500.0),
        _status(step_count=2),
        _telemetry(race_distance=200.0, speed_kph=1_500.0),
    )

    assert half_reference_speed.reward == pytest.approx(0.25)
    assert half_reference_speed.breakdown == {
        "frontier_progress": 1.0,
        "speed_progress": pytest.approx(-0.75),
    }
    assert half_top_speed_bonus.reward == pytest.approx(1.75)
    assert half_top_speed_bonus.breakdown == {
        "frontier_progress": 1.0,
        "speed_progress": pytest.approx(0.75),
    }


def test_reward_main_speed_curve_can_start_above_zero_speed() -> None:
    tracker = build_reward_tracker(
        RewardConfig(
            progress_bucket_distance=100.0,
            progress_bucket_reward=1.0,
            progress_speed_min_kph=500.0,
            progress_speed_min_multiplier=0.25,
            progress_speed_reference_kph=1_000.0,
            progress_speed_max_kph=2_000.0,
            progress_speed_max_multiplier=2.0,
            progress_speed_curve_power=1.0,
            time_penalty_per_frame=0.0,
            impact_frame_penalty=0.0,
        )
    )
    tracker.reset(_telemetry(race_distance=0.0, speed_kph=500.0))

    below_min_speed = tracker.step_summary(
        _summary(max_race_distance=100.0, max_race_distance_speed_kph=250.0),
        _status(step_count=1),
        _telemetry(race_distance=100.0, speed_kph=250.0),
    )
    halfway_to_reference = tracker.step_summary(
        _summary(max_race_distance=200.0, max_race_distance_speed_kph=750.0),
        _status(step_count=2),
        _telemetry(race_distance=200.0, speed_kph=750.0),
    )

    assert below_min_speed.reward == pytest.approx(0.25)
    assert below_min_speed.breakdown == {
        "frontier_progress": 1.0,
        "speed_progress": pytest.approx(-0.75),
    }
    assert halfway_to_reference.reward == pytest.approx(0.625)
    assert halfway_to_reference.breakdown == {
        "frontier_progress": 1.0,
        "speed_progress": pytest.approx(-0.375),
    }


def test_reward_main_scales_progress_by_race_position() -> None:
    tracker = build_reward_tracker(
        RewardConfig(
            progress_bucket_distance=100.0,
            progress_bucket_reward=1.0,
            position_progress_min_multiplier=0.9,
            position_progress_max_multiplier=1.2,
            time_penalty_per_frame=0.0,
            impact_frame_penalty=0.0,
        )
    )
    tracker.reset(_telemetry(race_distance=0.0, position=30))

    first_place = tracker.step_summary(
        _summary(max_race_distance=100.0),
        _status(step_count=1),
        _telemetry(race_distance=100.0, position=1),
    )
    last_place = tracker.step_summary(
        _summary(max_race_distance=200.0),
        _status(step_count=2),
        _telemetry(race_distance=200.0, position=30),
    )

    assert first_place.reward == pytest.approx(1.2)
    assert first_place.breakdown == {
        "frontier_progress": 1.0,
        "position_progress": pytest.approx(0.2),
    }
    assert last_place.reward == pytest.approx(0.9)
    assert last_place.breakdown == {
        "frontier_progress": 1.0,
        "position_progress": pytest.approx(-0.1),
    }


def test_reward_main_position_progress_multiplier_is_neutral_for_single_racer() -> None:
    tracker = build_reward_tracker(
        RewardConfig(
            progress_bucket_distance=100.0,
            progress_bucket_reward=1.0,
            position_progress_min_multiplier=0.8,
            position_progress_max_multiplier=1.2,
            time_penalty_per_frame=0.0,
            impact_frame_penalty=0.0,
        )
    )
    tracker.reset(_telemetry(race_distance=0.0, position=1, total_racers=1))

    reward = tracker.step_summary(
        _summary(max_race_distance=100.0),
        _status(step_count=1),
        _telemetry(race_distance=100.0, position=1, total_racers=1),
    )

    assert reward.reward == pytest.approx(1.0)
    assert reward.breakdown == {"frontier_progress": 1.0}


def test_reward_main_position_progress_uses_episode_start_racer_count() -> None:
    tracker = build_reward_tracker(
        RewardConfig(
            progress_bucket_distance=100.0,
            progress_bucket_reward=1.0,
            position_progress_min_multiplier=0.9,
            position_progress_max_multiplier=1.2,
            time_penalty_per_frame=0.0,
            impact_frame_penalty=0.0,
        )
    )
    tracker.reset(_telemetry(race_distance=0.0, position=30, total_racers=30))

    after_one_racer_dropped = tracker.step_summary(
        _summary(max_race_distance=100.0),
        _status(step_count=1),
        _telemetry(race_distance=100.0, position=29, total_racers=29),
    )

    assert after_one_racer_dropped.reward == pytest.approx(0.9103448276)
    assert after_one_racer_dropped.breakdown == {
        "frontier_progress": 1.0,
        "position_progress": pytest.approx(-0.0896551724),
    }


def test_reward_main_does_not_repay_skipped_progress_after_reentry() -> None:
    tracker = build_reward_tracker(
        RewardConfig(
            progress_bucket_distance=100.0,
            progress_bucket_reward=1.0,
            time_penalty_per_frame=0.0,
            impact_frame_penalty=0.0,
        )
    )
    tracker.reset(_telemetry(race_distance=0.0, current_radius_left=100.0))

    outside_airborne = tracker.step_summary(
        _summary(max_race_distance=1_800.0),
        _status(step_count=1),
        _telemetry(
            race_distance=1_800.0,
            state_labels=("active", "airborne"),
            signed_lateral_offset=150.0,
            current_radius_left=100.0,
        ),
    )
    inside_airborne = tracker.step_summary(
        _summary(max_race_distance=1_800.0),
        _status(step_count=2),
        _telemetry(
            race_distance=1_500.0,
            state_labels=("active", "airborne"),
            signed_lateral_offset=80.0,
            current_radius_left=100.0,
        ),
    )
    inside_grounded = tracker.step_summary(
        _summary(max_race_distance=1_800.0),
        _status(step_count=3),
        _telemetry(
            race_distance=1_300.0,
            signed_lateral_offset=80.0,
            current_radius_left=100.0,
        ),
    )

    assert outside_airborne.breakdown == {}
    assert inside_airborne.reward == 0.0
    assert inside_airborne.breakdown == {}
    assert inside_grounded.breakdown == {}


def test_reward_main_multiplies_frontier_progress_when_energy_refills() -> None:
    tracker = build_reward_tracker(
        RewardConfig(
            progress_bucket_distance=100.0,
            progress_bucket_reward=1.0,
            energy_refill_progress_multiplier=2.0,
            energy_gain_reward=0.0,
            time_penalty_per_frame=0.0,
            impact_frame_penalty=0.0,
        )
    )
    tracker.reset(_telemetry(race_distance=0.0, energy=89.0))

    step = tracker.step_summary(
        _summary(max_race_distance=100.0, energy_gain_total=10.0),
        _status(step_count=1),
        _telemetry(race_distance=100.0, energy=99.0, on_energy_refill=True),
    )

    assert step.reward == pytest.approx(2.0)
    assert step.breakdown["frontier_progress"] == 1.0
    assert step.breakdown["energy_refill_progress"] == pytest.approx(1.0)


def test_reward_main_penalizes_energy_loss_proportionally() -> None:
    tracker = build_reward_tracker(
        RewardConfig(
            progress_bucket_reward=0.0,
            time_penalty_per_frame=0.0,
            energy_loss_penalty=-0.02,
            impact_frame_penalty=0.0,
        )
    )
    tracker.reset(_telemetry(race_distance=0.0, energy=100.0))

    step = tracker.step_summary(
        _summary(max_race_distance=0.0, energy_loss_total=7.5),
        _status(step_count=1),
        _telemetry(race_distance=0.0, energy=92.5),
    )

    assert step.reward == pytest.approx(-0.15)
    assert step.breakdown == {"energy_loss": pytest.approx(-0.15)}


def test_reward_main_gates_energy_gain_reward_on_progress() -> None:
    tracker = build_reward_tracker(
        RewardConfig(
            progress_bucket_distance=100.0,
            progress_bucket_reward=1.0,
            energy_refill_progress_multiplier=1.0,
            energy_gain_reward=0.05,
            time_penalty_per_frame=0.0,
            impact_frame_penalty=0.0,
        )
    )
    tracker.reset(_telemetry(race_distance=0.0, energy=80.0))

    parked_refill = tracker.step_summary(
        _summary(max_race_distance=0.0, energy_gain_total=10.0),
        _status(step_count=1),
        _telemetry(race_distance=0.0, energy=90.0, on_energy_refill=True),
    )
    moving_refill = tracker.step_summary(
        _summary(max_race_distance=100.0, energy_gain_total=10.0),
        _status(step_count=2),
        _telemetry(race_distance=100.0, energy=100.0, on_energy_refill=True),
    )

    assert parked_refill.reward == 0.0
    assert parked_refill.breakdown == {}
    assert moving_refill.reward == pytest.approx(1.5)
    assert moving_refill.breakdown == {
        "frontier_progress": 1.0,
        "energy_gain": pytest.approx(0.5),
    }


def test_reward_main_keeps_manual_boost_reward_constant_when_energy_shaping_is_off() -> None:
    tracker = build_reward_tracker(
        RewardConfig(
            progress_bucket_reward=0.0,
            time_penalty_per_frame=0.0,
            impact_frame_penalty=0.0,
            manual_boost_reward=8.0,
            manual_boost_reward_energy_shaping=False,
        )
    )
    tracker.reset(_telemetry(energy=178.0))

    step = tracker.step_summary(
        _summary(max_race_distance=0.0),
        _status(step_count=1),
        _telemetry(energy=44.5),
        RewardActionContext(boost_requested=True),
    )

    assert step.reward == pytest.approx(8.0)
    assert step.breakdown == {"manual_boost": pytest.approx(8.0)}


def test_reward_main_scales_manual_boost_reward_by_energy_fraction() -> None:
    tracker = build_reward_tracker(
        RewardConfig(
            progress_bucket_reward=0.0,
            time_penalty_per_frame=0.0,
            impact_frame_penalty=0.0,
            manual_boost_reward=8.0,
            manual_boost_reward_energy_shaping=True,
            manual_boost_reward_min_energy_fraction=0.0,
            manual_boost_reward_min_energy_value=2.0,
            manual_boost_reward_full_energy_fraction=1.0,
            manual_boost_reward_energy_curve="linear",
        )
    )
    tracker.reset(_telemetry(energy=178.0))

    step = tracker.step_summary(
        _summary(max_race_distance=0.0),
        _status(step_count=1),
        _telemetry(energy=89.0),
        RewardActionContext(boost_requested=True),
    )

    assert step.reward == pytest.approx(5.0)
    assert step.breakdown == {"manual_boost": pytest.approx(5.0)}


def test_reward_main_holds_manual_boost_reward_at_low_energy_floor() -> None:
    tracker = build_reward_tracker(
        RewardConfig(
            progress_bucket_reward=0.0,
            time_penalty_per_frame=0.0,
            impact_frame_penalty=0.0,
            manual_boost_reward=8.0,
            manual_boost_reward_energy_shaping=True,
            manual_boost_reward_min_energy_fraction=0.5,
            manual_boost_reward_min_energy_value=2.0,
            manual_boost_reward_full_energy_fraction=1.0,
            manual_boost_reward_energy_curve="linear",
        )
    )
    tracker.reset(_telemetry(energy=178.0))

    floor_step = tracker.step_summary(
        _summary(max_race_distance=0.0),
        _status(step_count=1),
        _telemetry(energy=44.5),
        RewardActionContext(boost_requested=True),
    )
    ramp_step = tracker.step_summary(
        _summary(max_race_distance=0.0),
        _status(step_count=2),
        _telemetry(energy=133.5),
        RewardActionContext(boost_requested=True),
    )

    assert floor_step.reward == pytest.approx(2.0)
    assert floor_step.breakdown == {"manual_boost": pytest.approx(2.0)}
    assert ramp_step.reward == pytest.approx(5.0)
    assert ramp_step.breakdown == {"manual_boost": pytest.approx(5.0)}


def test_reward_main_allows_low_energy_boost_request_penalty() -> None:
    tracker = build_reward_tracker(
        RewardConfig(
            progress_bucket_reward=0.0,
            time_penalty_per_frame=0.0,
            impact_frame_penalty=0.0,
            manual_boost_reward=8.0,
            manual_boost_reward_energy_shaping=True,
            manual_boost_reward_min_energy_fraction=0.5,
            manual_boost_reward_min_energy_value=-4.0,
            manual_boost_reward_full_energy_fraction=1.0,
            manual_boost_reward_energy_curve="linear",
        )
    )
    tracker.reset(_telemetry(energy=178.0))

    floor_step = tracker.step_summary(
        _summary(max_race_distance=0.0),
        _status(step_count=1),
        _telemetry(energy=44.5),
        RewardActionContext(boost_requested=True),
    )
    ramp_step = tracker.step_summary(
        _summary(max_race_distance=0.0),
        _status(step_count=2),
        _telemetry(energy=133.5),
        RewardActionContext(boost_requested=True),
    )

    assert floor_step.reward == pytest.approx(-4.0)
    assert floor_step.breakdown == {"manual_boost": pytest.approx(-4.0)}
    assert ramp_step.reward == pytest.approx(2.0)
    assert ramp_step.breakdown == {"manual_boost": pytest.approx(2.0)}


def test_reward_main_suppresses_refill_multiplier_at_full_energy() -> None:
    tracker = build_reward_tracker(
        RewardConfig(
            progress_bucket_distance=100.0,
            progress_bucket_reward=1.0,
            energy_refill_progress_multiplier=2.0,
            time_penalty_per_frame=0.0,
            impact_frame_penalty=0.0,
        )
    )
    tracker.reset(_telemetry(race_distance=0.0, energy=178.0))

    step = tracker.step_summary(
        _summary(max_race_distance=100.0),
        _status(step_count=1),
        _telemetry(race_distance=100.0, energy=178.0, on_energy_refill=True),
    )

    assert step.reward == pytest.approx(1.0)
    assert step.breakdown == {"frontier_progress": 1.0}


def test_reward_main_scales_frontier_progress_on_dirt_and_ice() -> None:
    tracker = build_reward_tracker(
        RewardConfig(
            progress_bucket_distance=100.0,
            progress_bucket_reward=1.0,
            dirt_progress_multiplier=0.5,
            ice_progress_multiplier=0.7,
            time_penalty_per_frame=0.0,
            impact_frame_penalty=0.0,
        )
    )
    tracker.reset(_telemetry(race_distance=0.0))

    dirt_step = tracker.step_summary(
        _summary(max_race_distance=100.0),
        _status(step_count=1),
        _telemetry(race_distance=100.0, course_effect_raw=_COURSE_EFFECT_DIRT),
    )
    ice_step = tracker.step_summary(
        _summary(max_race_distance=200.0),
        _status(step_count=2),
        _telemetry(race_distance=200.0, course_effect_raw=_COURSE_EFFECT_ICE),
    )

    assert dirt_step.breakdown == {"frontier_progress": 1.0, "dirt_progress": -0.5}
    assert ice_step.breakdown == {
        "frontier_progress": 1.0,
        "ice_progress": pytest.approx(-0.3),
    }


def test_reward_main_penalizes_bad_ground_entry_once_per_transition() -> None:
    tracker = build_reward_tracker(
        RewardConfig(
            progress_bucket_reward=0.0,
            time_penalty_per_frame=0.0,
            dirt_entry_penalty=-0.5,
            ice_entry_penalty=-0.25,
            impact_frame_penalty=0.0,
        )
    )
    tracker.reset(_telemetry(race_distance=0.0))

    dirt_entry = tracker.step_summary(
        _summary(max_race_distance=0.0),
        _status(step_count=1),
        _telemetry(race_distance=0.0, course_effect_raw=_COURSE_EFFECT_DIRT),
    )
    still_on_dirt = tracker.step_summary(
        _summary(max_race_distance=0.0),
        _status(step_count=2),
        _telemetry(race_distance=0.0, course_effect_raw=_COURSE_EFFECT_DIRT),
    )
    ice_entry = tracker.step_summary(
        _summary(max_race_distance=0.0),
        _status(step_count=3),
        _telemetry(race_distance=0.0, course_effect_raw=_COURSE_EFFECT_ICE),
    )

    assert dirt_entry.breakdown == {"dirt_entry": -0.5}
    assert still_on_dirt.breakdown == {}
    assert ice_entry.breakdown == {"ice_entry": -0.25}


def test_reward_main_rewards_dash_pad_boost_entries_once_per_progress_window() -> None:
    tracker = build_reward_tracker(
        RewardConfig(
            progress_bucket_distance=100.0,
            progress_bucket_reward=1.0,
            boost_pad_reward=0.5,
            boost_pad_reward_progress_window=1_000.0,
            time_penalty_per_frame=0.0,
            impact_frame_penalty=0.0,
        )
    )
    tracker.reset(_telemetry(race_distance=0.0))

    first = tracker.step_summary(
        _summary(
            max_race_distance=100.0,
            entered_course_effects=_entered_course_effects(_COURSE_EFFECT_DASH),
        ),
        _status(step_count=1),
        _telemetry(race_distance=100.0, course_effect_raw=_COURSE_EFFECT_DASH),
    )
    blocked_same_window = tracker.step_summary(
        _summary(
            max_race_distance=900.0,
            entered_course_effects=_entered_course_effects(_COURSE_EFFECT_DASH),
        ),
        _status(step_count=2),
        _telemetry(race_distance=900.0, course_effect_raw=_COURSE_EFFECT_DASH),
    )
    rewarded_next_window = tracker.step_summary(
        _summary(
            max_race_distance=1_100.0,
            entered_course_effects=_entered_course_effects(_COURSE_EFFECT_DASH),
        ),
        _status(step_count=3),
        _telemetry(race_distance=1_100.0, course_effect_raw=_COURSE_EFFECT_DASH),
    )

    assert first.breakdown["boost_pad"] == 0.5
    assert "boost_pad" not in blocked_same_window.breakdown
    assert rewarded_next_window.breakdown["boost_pad"] == 0.5


def test_reward_main_uses_course_reward_override() -> None:
    config = RewardConfig(
        progress_bucket_reward=1.0,
        time_penalty_per_frame=0.0,
        course_overrides={
            "mute-city-i": RewardCourseOverrideConfig(progress_bucket_reward=2.0),
        },
    )
    tracker = build_reward_tracker(config)
    tracker.reset(_telemetry(race_distance=0.0), course_id="mute-city-i")

    step = tracker.step_summary(
        _summary(max_race_distance=1_000.0),
        _status(step_count=1),
        _telemetry(race_distance=1_000.0),
    )

    assert step.breakdown == {"frontier_progress": 2.0}


def test_reward_main_course_reward_override_null_fields_inherit_base_after_dump() -> None:
    config = RewardConfig(
        time_penalty_per_frame=-0.01,
        course_overrides={
            "mute-city-i": RewardCourseOverrideConfig(progress_bucket_reward=2.0),
        },
    )
    tracker = build_reward_tracker(config)
    tracker.reset(_telemetry(race_distance=0.0), course_id="mute-city-i")

    step = tracker.step_summary(
        _summary(max_race_distance=0.0, frames_run=2),
        _status(step_count=1),
        _telemetry(race_distance=0.0),
    )

    assert step.breakdown == {"time": pytest.approx(-0.02)}


def _telemetry(
    *,
    race_distance: float = 0.0,
    game_mode_name: str = "gp_race",
    state_labels: tuple[str, ...] = ("active",),
    total_racers: int = 30,
    position: int = 30,
    energy: float = 178.0,
    ko_star_count: int = 0,
    boost_timer: int = 0,
    race_time_ms: int = 0,
    speed_kph: float = 100.0,
    laps_completed: int = 0,
    lap: int | None = None,
    reverse_timer: int = 0,
    on_energy_refill: bool = False,
    course_effect_raw: int = 0,
    height_above_ground: float = 0.0,
    signed_lateral_offset: float = 0.0,
    lateral_distance: float = 0.0,
    current_radius_left: float = 0.0,
    current_radius_right: float = 0.0,
    future_local_nearest_segment_index: int | None = None,
    future_local_nearest_segment_distance: float = 0.0,
) -> FZeroXTelemetry:
    state_flags = encode_state_flags(state_labels)
    state_flags |= course_effect_raw
    if on_energy_refill:
        state_flags |= _COURSE_EFFECT_PIT
    return make_telemetry(
        game_mode_name=game_mode_name,
        total_racers=total_racers,
        race_distance=race_distance,
        state_labels=state_labels,
        state_flags=state_flags,
        speed_kph=speed_kph,
        energy=energy,
        ko_star_count=ko_star_count,
        boost_timer=boost_timer,
        race_time_ms=race_time_ms,
        position=position,
        laps_completed=laps_completed,
        lap=max(laps_completed + 1, 1) if lap is None else lap,
        reverse_timer=reverse_timer,
        height_above_ground=height_above_ground,
        signed_lateral_offset=signed_lateral_offset,
        lateral_distance=lateral_distance,
        current_radius_left=current_radius_left,
        current_radius_right=current_radius_right,
        future_local_nearest_segment_index=future_local_nearest_segment_index,
        future_local_nearest_segment_distance=future_local_nearest_segment_distance,
    )


def _summary(
    *,
    max_race_distance: float,
    max_race_distance_speed_kph: float = 760.0,
    frames_run: int = 1,
    airborne_frames: int = 0,
    reverse_active_frames: int = 0,
    low_speed_frames: int = 0,
    energy_loss_total: float = 0.0,
    energy_gain_total: float = 0.0,
    damage_taken_frames: int = 0,
    entered_state_labels: tuple[str, ...] = (),
    entered_course_effects: int = 0,
) -> StepSummary:
    return make_step_summary(
        frames_run=frames_run,
        max_race_distance=max_race_distance,
        max_race_distance_speed_kph=max_race_distance_speed_kph,
        reverse_active_frames=reverse_active_frames,
        low_speed_frames=low_speed_frames,
        energy_loss_total=energy_loss_total,
        energy_gain_total=energy_gain_total,
        damage_taken_frames=damage_taken_frames,
        entered_state_labels=entered_state_labels,
        entered_course_effects=entered_course_effects,
        final_frame_index=frames_run,
        airborne_frames=airborne_frames,
    )


def _status(
    *,
    step_count: int,
    stalled_steps: int = 0,
    reverse_timer: int = 0,
    termination_reason: str | None = None,
    truncation_reason: str | None = None,
) -> StepStatus:
    return StepStatus(
        {
            "step_count": step_count,
            "stalled_steps": stalled_steps,
            "reverse_timer": reverse_timer,
            "termination_reason": termination_reason,
            "truncation_reason": truncation_reason,
        }
    )


def _entered_course_effects(*effects: int) -> int:
    bitset = 0
    for effect in effects:
        bitset |= 1 << effect
    return bitset
