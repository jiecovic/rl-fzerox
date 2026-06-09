# tests/core/game/test_reward_progress.py
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
    assert gained_two.debug_info == {
        "ko_star_reward_event": True,
        "ko_star_reward_previous_count": 1,
        "ko_star_reward_current_count": 3,
        "ko_star_reward_gain": 2,
        "ko_star_reward_value": 5.0,
    }
    assert repeated_count.reward == 0.0
    assert repeated_count.debug_info == {}


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
    assert step.debug_info == {}


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


def test_reward_main_can_pay_continuous_frontier_progress() -> None:
    tracker = build_reward_tracker(
        RewardConfig(
            progress_bucket_distance=0.0,
            progress_bucket_reward=2.0,
            time_penalty_per_frame=0.0,
            impact_frame_penalty=0.0,
        )
    )
    tracker.reset(_telemetry(race_distance=0.0))

    first_step = tracker.step_summary(
        _summary(max_race_distance=2_500.0),
        _status(step_count=1),
        _telemetry(race_distance=2_500.0),
    )
    repeated_step = tracker.step_summary(
        _summary(max_race_distance=2_500.0),
        _status(step_count=2),
        _telemetry(race_distance=500.0),
    )

    assert first_step.reward == 5.0
    assert first_step.breakdown == {"frontier_progress": 5.0}
    assert repeated_step.reward == 0.0
    info = tracker.info(_telemetry(race_distance=2_500.0))
    assert info["frontier_progress_distance"] == 2_500.0
    assert info["progress_bucket_distance"] == 0.0
    assert info["progress_bucket_reward"] == 2.0


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
