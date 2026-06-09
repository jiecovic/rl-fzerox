# tests/core/game/test_reward_energy.py
from __future__ import annotations

import pytest

from rl_fzerox.core.envs.rewards import (
    RewardActionContext,
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
