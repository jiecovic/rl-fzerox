# tests/core/game/test_reward.py
from __future__ import annotations

import pytest

from fzerox_emulator import FZeroXTelemetry, StepStatus, StepSummary
from rl_fzerox.core.envs.rewards import (
    DEFAULT_REWARD_NAME,
    REWARD_TRACKER_REGISTRY,
    RaceV2RewardTracker,
    RaceV2RewardWeights,
    build_reward_tracker,
    reward_tracker_names,
)
from tests.support.native_objects import make_step_summary, make_telemetry


def test_race_v2_rewards_each_progress_milestone_once() -> None:
    tracker = RaceV2RewardTracker(
        RaceV2RewardWeights(
            time_penalty_per_frame=0.0,
            milestone_distance=25.0,
            milestone_bonus=3.0,
        )
    )
    tracker.reset(_telemetry(race_distance=0.0))

    first = tracker.step_summary(
        _summary(max_race_distance=60.0),
        _status(step_count=1),
        _telemetry(race_distance=60.0),
    )
    repeated = tracker.step_summary(
        _summary(max_race_distance=70.0),
        _status(step_count=2),
        _telemetry(race_distance=70.0),
    )

    assert first.reward == pytest.approx(6.0)
    assert first.breakdown == {"milestone": 6.0}
    assert repeated.reward == 0.0
    assert repeated.breakdown == {}


def test_race_v2_rewards_completed_laps() -> None:
    tracker = RaceV2RewardTracker(
        RaceV2RewardWeights(
            time_penalty_per_frame=0.0,
            milestone_bonus=0.0,
            lap_1_completion_bonus=20.0,
            lap_2_completion_bonus=35.0,
            final_lap_completion_bonus=60.0,
            lap_position_scale=1.0,
        )
    )
    tracker.reset(_telemetry(race_distance=0.0, laps_completed=0))

    step = tracker.step_summary(
        _summary(max_race_distance=80_000.0),
        _status(step_count=200),
        _telemetry(race_distance=80_000.0, laps_completed=1, position=1),
    )

    assert step.reward == pytest.approx(49.0)
    assert step.breakdown == {
        "lap_completion": 20.0,
        "lap_position": 29.0,
    }


def test_race_v2_scales_time_penalty_while_reverse_warning_is_active() -> None:
    tracker = RaceV2RewardTracker(
        RaceV2RewardWeights(
            time_penalty_per_frame=-0.01,
            reverse_time_penalty_scale=2.0,
            milestone_bonus=0.0,
        )
    )
    tracker.reset(_telemetry(race_distance=0.0))

    step = tracker.step_summary(
        _summary(max_race_distance=0.0, frames_run=3, reverse_warning_frames=2),
        _status(step_count=3, reverse_timer=120),
        _telemetry(race_distance=0.0, reverse_timer=120),
    )

    assert step.reward == pytest.approx(-0.05)
    assert step.breakdown == {
        "time": -0.03,
        "reverse_time": -0.02,
    }


def test_race_v2_penalizes_energy_loss_more_than_refill_reward() -> None:
    tracker = RaceV2RewardTracker(
        RaceV2RewardWeights(
            time_penalty_per_frame=0.0,
            energy_loss_epsilon=0.1,
            energy_loss_penalty_scale=0.05,
            energy_gain_reward_scale=0.02,
        )
    )
    tracker.reset(_telemetry(race_distance=100.0, energy=178.0))

    loss = tracker.step_summary(
        _summary(max_race_distance=100.0, energy_loss_total=4.0),
        _status(step_count=1),
        _telemetry(race_distance=100.0, energy=174.0),
    )
    gain = tracker.step_summary(
        _summary(max_race_distance=100.0, energy_gain_total=2.0),
        _status(step_count=2),
        _telemetry(race_distance=100.0, energy=176.0),
    )

    assert loss.reward == -0.2
    assert loss.breakdown == {"energy_loss": -0.2}
    assert gain.reward == pytest.approx(0.04)
    assert gain.breakdown == {"energy_gain": 0.04}


def test_race_v2_energy_gain_stays_net_negative_against_equal_loss() -> None:
    tracker = RaceV2RewardTracker(
        RaceV2RewardWeights(
            time_penalty_per_frame=0.0,
            energy_loss_penalty_scale=0.05,
            energy_gain_reward_scale=0.02,
        )
    )
    tracker.reset(_telemetry(race_distance=100.0, energy=178.0))

    step = tracker.step_summary(
        _summary(
            max_race_distance=100.0,
            energy_loss_total=4.0,
            energy_gain_total=4.0,
        ),
        _status(step_count=1),
        _telemetry(race_distance=100.0, energy=178.0),
    )

    assert step.reward == pytest.approx(-0.12)
    assert step.breakdown == {
        "energy_loss": -0.2,
        "energy_gain": 0.08,
    }


def test_race_v2_applies_event_penalties_once_per_entry() -> None:
    tracker = RaceV2RewardTracker(
        RaceV2RewardWeights(
            time_penalty_per_frame=0.0,
            collision_recoil_penalty=-2.0,
        )
    )
    tracker.reset(_telemetry(race_distance=100.0))

    first = tracker.step_summary(
        _summary(
            max_race_distance=100.0,
            entered_state_labels=("collision_recoil",),
        ),
        _status(step_count=1),
        _telemetry(
            race_distance=100.0,
            state_labels=("active", "collision_recoil"),
        ),
    )
    repeated = tracker.step_summary(
        _summary(max_race_distance=100.0),
        _status(step_count=2),
        _telemetry(
            race_distance=100.0,
            state_labels=("active", "collision_recoil"),
        ),
    )

    assert first.reward == -2.0
    assert first.breakdown == {"collision_recoil": -2.0}
    assert repeated.reward == 0.0
    assert repeated.breakdown == {}


def test_race_v2_scales_terminal_failure_penalty_by_remaining_steps_and_laps() -> None:
    tracker = RaceV2RewardTracker(
        RaceV2RewardWeights(
            time_penalty_per_frame=-0.01,
            milestone_bonus=0.0,
            lap_1_completion_bonus=0.0,
            lap_2_completion_bonus=0.0,
            final_lap_completion_bonus=0.0,
            lap_position_scale=0.0,
            remaining_lap_penalty=50.0,
            terminal_failure_base_penalty=-120.0,
        ),
        max_episode_steps=100,
    )
    tracker.reset(_telemetry(race_distance=0.0, laps_completed=0))

    early = tracker.step_summary(
        _summary(max_race_distance=10_000.0),
        _status(step_count=10, termination_reason="crashed"),
        _telemetry(race_distance=10_000.0, laps_completed=0, state_labels=("crashed",)),
    )
    late = tracker.step_summary(
        _summary(max_race_distance=150_000.0),
        _status(step_count=90, termination_reason="crashed"),
        _telemetry(race_distance=150_000.0, laps_completed=2, state_labels=("crashed",)),
    )

    assert early.reward == pytest.approx(-270.91)
    assert early.breakdown["crashed"] == pytest.approx(-270.9)
    assert late.reward == pytest.approx(-170.11)
    assert late.breakdown["crashed"] == pytest.approx(-170.1)
    assert early.reward < late.reward


def test_race_v2_scales_truncation_penalty_by_remaining_steps_and_laps() -> None:
    tracker = RaceV2RewardTracker(
        RaceV2RewardWeights(
            time_penalty_per_frame=-0.01,
            milestone_bonus=0.0,
            lap_1_completion_bonus=0.0,
            lap_2_completion_bonus=0.0,
            final_lap_completion_bonus=0.0,
            lap_position_scale=0.0,
            remaining_lap_penalty=50.0,
            wrong_way_truncation_base_penalty=-170.0,
        ),
        max_episode_steps=100,
    )
    tracker.reset(_telemetry(race_distance=0.0, laps_completed=0))

    early = tracker.step_summary(
        _summary(max_race_distance=0.0),
        _status(step_count=10, reverse_timer=100, truncation_reason="wrong_way"),
        _telemetry(race_distance=0.0, laps_completed=0),
    )
    late = tracker.step_summary(
        _summary(max_race_distance=140_000.0),
        _status(step_count=90, reverse_timer=100, truncation_reason="wrong_way"),
        _telemetry(race_distance=140_000.0, laps_completed=2),
    )

    assert early.breakdown["wrong_way_truncation"] == pytest.approx(-320.9)
    assert late.breakdown["wrong_way_truncation"] == pytest.approx(-220.1)
    assert early.reward == pytest.approx(-320.91)
    assert late.reward == pytest.approx(-220.11)
    assert early.reward < late.reward


def test_race_v2_applies_final_lap_reward_and_finish_position_bonus() -> None:
    tracker = RaceV2RewardTracker(
        RaceV2RewardWeights(
            time_penalty_per_frame=-0.01,
            milestone_bonus=0.0,
            lap_1_completion_bonus=20.0,
            lap_2_completion_bonus=35.0,
            final_lap_completion_bonus=60.0,
            lap_position_scale=0.0,
            finish_position_scale=4.0,
        )
    )
    tracker.reset(_telemetry(race_distance=0.0, state_labels=("active",), laps_completed=2))

    step = tracker.step_summary(
        _summary(max_race_distance=100.0, entered_state_labels=("finished",)),
        _status(step_count=120, termination_reason="finished"),
        _telemetry(
            race_distance=100.0,
            state_labels=("active", "finished"),
            position=1,
            laps_completed=3,
        ),
    )

    assert step.reward == pytest.approx(175.99)
    assert step.breakdown == {
        "time": -0.01,
        "lap_completion": 60.0,
        "finish_position": 116.0,
    }


def test_race_v2_keeps_low_speed_and_dash_pad_out_of_reward() -> None:
    tracker = RaceV2RewardTracker(
        RaceV2RewardWeights(
            time_penalty_per_frame=-0.01,
        )
    )
    tracker.reset(_telemetry(race_distance=100.0, speed_kph=30.0))

    step = tracker.step_summary(
        _summary(
            max_race_distance=100.0,
            entered_state_labels=("dash_pad_boost",),
        ),
        _status(step_count=1),
        _telemetry(
            race_distance=100.0,
            speed_kph=30.0,
            state_labels=("active", "dash_pad_boost"),
        ),
    )

    assert step.reward == -0.01
    assert step.breakdown == {"time": -0.01}


def test_reward_tracker_registry_exposes_registered_names() -> None:
    assert DEFAULT_REWARD_NAME == "race_v2"
    assert reward_tracker_names() == tuple(REWARD_TRACKER_REGISTRY)
    assert isinstance(build_reward_tracker(), RaceV2RewardTracker)


def test_race_v2_multiplies_time_penalty_by_frames_run() -> None:
    tracker = RaceV2RewardTracker(
        RaceV2RewardWeights(
            time_penalty_per_frame=-0.01,
        )
    )
    tracker.reset(_telemetry(race_distance=100.0))

    step = tracker.step_summary(
        _summary(max_race_distance=100.0, frames_run=3),
        _status(step_count=3),
        _telemetry(race_distance=100.0),
    )

    assert step.reward == pytest.approx(-0.03)
    assert step.breakdown == {"time": -0.03}


def _telemetry(
    *,
    race_distance: float,
    state_labels: tuple[str, ...] = ("active",),
    position: int = 30,
    energy: float = 178.0,
    boost_timer: int = 0,
    race_time_ms: int = 0,
    speed_kph: float = 100.0,
    laps_completed: int = 0,
    reverse_timer: int = 0,
) -> FZeroXTelemetry:
    return make_telemetry(
        race_distance=race_distance,
        state_labels=state_labels,
        speed_kph=speed_kph,
        energy=energy,
        boost_timer=boost_timer,
        race_time_ms=race_time_ms,
        position=position,
        laps_completed=laps_completed,
        reverse_timer=reverse_timer,
    )


def _summary(
    *,
    max_race_distance: float,
    frames_run: int = 1,
    reverse_warning_frames: int = 0,
    energy_loss_total: float = 0.0,
    energy_gain_total: float = 0.0,
    entered_state_labels: tuple[str, ...] = (),
) -> StepSummary:
    return make_step_summary(
        frames_run=frames_run,
        max_race_distance=max_race_distance,
        reverse_warning_frames=reverse_warning_frames,
        energy_loss_total=energy_loss_total,
        energy_gain_total=energy_gain_total,
        entered_state_labels=entered_state_labels,
        final_frame_index=frames_run,
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
        step_count=step_count,
        stalled_steps=stalled_steps,
        reverse_timer=reverse_timer,
        termination_reason=termination_reason,
        truncation_reason=truncation_reason,
    )
