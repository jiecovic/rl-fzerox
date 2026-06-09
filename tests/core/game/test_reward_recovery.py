# tests/core/game/test_reward_recovery.py
from __future__ import annotations

import pytest

from rl_fzerox.core.envs.rewards import (
    build_reward_tracker,
)
from rl_fzerox.core.runtime_spec.schema import RewardConfig
from tests.core.game.reward_support import (
    _status,
    _summary,
    _telemetry,
)

_COURSE_EFFECT_PIT = 1
_COURSE_EFFECT_DIRT = 2
_COURSE_EFFECT_DASH = 3
_COURSE_EFFECT_ICE = 4


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
