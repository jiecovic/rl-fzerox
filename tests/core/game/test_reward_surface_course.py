# tests/core/game/test_reward_surface_course.py
from __future__ import annotations

import pytest

from rl_fzerox.core.envs.rewards import (
    build_reward_tracker,
)
from rl_fzerox.core.runtime_spec.schema import RewardConfig, RewardCourseOverrideConfig
from tests.core.game.reward_support import (
    _entered_course_effects,
    _status,
    _summary,
    _telemetry,
)

_COURSE_EFFECT_PIT = 1
_COURSE_EFFECT_DIRT = 2
_COURSE_EFFECT_DASH = 3
_COURSE_EFFECT_ICE = 4


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


def test_reward_main_rewards_dash_pad_boost_entries_once_per_progress_window() -> None:
    tracker = build_reward_tracker(
        RewardConfig(
            progress_bucket_distance=100.0,
            progress_bucket_reward=1.0,
            boost_pad_reward_before_unlock=0.5,
            boost_pad_reward_after_unlock=0.5,
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


def test_reward_main_splits_dash_pad_reward_by_manual_boost_unlock() -> None:
    tracker = build_reward_tracker(
        RewardConfig(
            progress_bucket_distance=100.0,
            progress_bucket_reward=1.0,
            boost_pad_reward_before_unlock=1.25,
            boost_pad_reward_after_unlock=0.25,
            boost_pad_reward_progress_window=100.0,
            time_penalty_per_frame=0.0,
            impact_frame_penalty=0.0,
        )
    )
    tracker.reset(_telemetry(race_distance=0.0))

    before_unlock = tracker.step_summary(
        _summary(
            max_race_distance=100.0,
            entered_course_effects=_entered_course_effects(_COURSE_EFFECT_DASH),
        ),
        _status(step_count=1),
        _telemetry(race_distance=100.0, course_effect_raw=_COURSE_EFFECT_DASH),
    )
    after_unlock = tracker.step_summary(
        _summary(
            max_race_distance=200.0,
            entered_course_effects=_entered_course_effects(_COURSE_EFFECT_DASH),
        ),
        _status(step_count=2),
        _telemetry(
            race_distance=200.0,
            course_effect_raw=_COURSE_EFFECT_DASH,
            state_labels=("active", "can_boost"),
        ),
    )

    assert before_unlock.breakdown["boost_pad"] == 1.25
    assert after_unlock.breakdown["boost_pad"] == 0.25


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
