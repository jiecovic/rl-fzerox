# tests/core/game/test_reward.py
from __future__ import annotations

import pytest

from rl_fzerox.core.emulator import StepSummary
from rl_fzerox.core.envs.rewards import (
    DEFAULT_REWARD_NAME,
    REWARD_TRACKER_REGISTRY,
    RaceV2RewardTracker,
    RaceV2RewardWeights,
    build_reward_tracker,
    reward_tracker_names,
)
from rl_fzerox.core.game import FZeroXTelemetry, PlayerTelemetry
from rl_fzerox.core.game.flags import (
    FLAG_ACTIVE,
    FLAG_COLLISION_RECOIL,
    FLAG_DASH_PAD_BOOST,
    FLAG_FINISHED,
)


def test_race_v2_rewards_new_frontier_progress_once() -> None:
    tracker = RaceV2RewardTracker(
        RaceV2RewardWeights(
            progress_scale=0.001,
            time_penalty_per_frame=-0.01,
        )
    )
    tracker.reset(_telemetry(race_distance=100.0))

    first = tracker.step_summary(
        _summary(max_race_distance=130.0),
        _telemetry(race_distance=130.0),
    )
    second = tracker.step_summary(
        _summary(max_race_distance=120.0, reverse_progress_total=10.0),
        _telemetry(race_distance=120.0),
    )

    assert first.reward == pytest.approx(0.02)
    assert first.breakdown == {"time": -0.01, "progress": 0.03}
    assert second.reward == pytest.approx(-0.02)
    assert second.breakdown == {"time": -0.01, "reverse_progress": -0.01}


def test_race_v2_ignores_small_progress_noise() -> None:
    tracker = RaceV2RewardTracker(
        RaceV2RewardWeights(
            progress_scale=0.001,
            progress_epsilon=0.5,
            time_penalty_per_frame=-0.01,
        )
    )
    tracker.reset(_telemetry(race_distance=100.0))

    step = tracker.step_summary(
        _summary(max_race_distance=100.2),
        _telemetry(race_distance=100.2),
    )

    assert step.reward == -0.01
    assert step.breakdown == {"time": -0.01}


def test_race_v2_penalizes_energy_loss_but_not_energy_gain() -> None:
    tracker = RaceV2RewardTracker(
        RaceV2RewardWeights(
            progress_scale=0.0,
            time_penalty_per_frame=0.0,
            energy_loss_epsilon=0.1,
            energy_loss_penalty_scale=0.05,
        )
    )
    tracker.reset(_telemetry(race_distance=100.0, energy=178.0))

    loss = tracker.step_summary(
        _summary(max_race_distance=100.0, energy_loss_total=4.0),
        _telemetry(race_distance=100.0, energy=174.0),
    )
    gain = tracker.step_summary(
        _summary(max_race_distance=100.0, energy_loss_total=0.0),
        _telemetry(race_distance=100.0, energy=176.0),
    )

    assert loss.reward == -0.2
    assert loss.breakdown == {"energy_loss": -0.2}
    assert gain.reward == 0.0
    assert gain.breakdown == {}


def test_race_v2_applies_event_penalties_once_per_entry() -> None:
    tracker = RaceV2RewardTracker(
        RaceV2RewardWeights(
            progress_scale=0.0,
            time_penalty_per_frame=0.0,
            collision_recoil_penalty=-2.0,
        )
    )
    tracker.reset(_telemetry(race_distance=100.0))

    first = tracker.step_summary(
        _summary(max_race_distance=100.0, entered_state_flags=FLAG_COLLISION_RECOIL),
        _telemetry(race_distance=100.0, state_flags=FLAG_ACTIVE | FLAG_COLLISION_RECOIL),
    )
    repeated = tracker.step_summary(
        _summary(max_race_distance=100.0),
        _telemetry(race_distance=100.0, state_flags=FLAG_ACTIVE | FLAG_COLLISION_RECOIL),
    )

    assert first.reward == -2.0
    assert first.breakdown == {"collision_recoil": -2.0}
    assert repeated.reward == 0.0
    assert repeated.breakdown == {}


def test_race_v2_applies_finish_bonus_and_position_bonus() -> None:
    tracker = RaceV2RewardTracker(
        RaceV2RewardWeights(
            progress_scale=0.001,
            time_penalty_per_frame=-0.01,
            finish_bonus=150.0,
            finish_position_scale=4.0,
        )
    )
    tracker.reset(_telemetry(race_distance=0.0, state_flags=FLAG_ACTIVE))

    step = tracker.step_summary(
        _summary(max_race_distance=100.0, entered_state_flags=FLAG_FINISHED),
        _telemetry(
            race_distance=100.0,
            state_flags=FLAG_ACTIVE | FLAG_FINISHED,
            position=1,
        )
    )

    assert step.terminated is True
    assert step.reward == pytest.approx(266.09)
    assert step.breakdown == {
        "time": -0.01,
        "progress": 0.1,
        "finished": 150.0,
        "finish_position": 116.0,
    }


def test_race_v2_keeps_low_speed_and_dash_pad_out_of_reward() -> None:
    tracker = RaceV2RewardTracker(
        RaceV2RewardWeights(
            progress_scale=0.0,
            time_penalty_per_frame=-0.01,
        )
    )
    tracker.reset(_telemetry(race_distance=100.0, speed_kph=30.0))

    step = tracker.step_summary(
        _summary(max_race_distance=100.0, entered_state_flags=FLAG_DASH_PAD_BOOST),
        _telemetry(
            race_distance=100.0,
            speed_kph=30.0,
            state_flags=FLAG_ACTIVE | FLAG_DASH_PAD_BOOST,
        )
    )

    assert step.reward == -0.01
    assert step.breakdown == {"time": -0.01}


def test_race_v2_returns_truncation_penalties() -> None:
    tracker = RaceV2RewardTracker(
        RaceV2RewardWeights(
            stuck_truncation_penalty=-100.0,
            wrong_way_truncation_penalty=-120.0,
            timeout_truncation_penalty=-80.0,
        )
    )

    assert tracker.truncation_penalty("stuck") == (-100.0, "stuck_truncation")
    assert tracker.truncation_penalty("wrong_way") == (-120.0, "wrong_way_truncation")
    assert tracker.truncation_penalty("timeout") == (-80.0, "timeout_truncation")
    assert tracker.truncation_penalty(None) == (0.0, None)


def test_reward_tracker_registry_exposes_registered_names() -> None:
    assert DEFAULT_REWARD_NAME == "race_v2"
    assert reward_tracker_names() == tuple(REWARD_TRACKER_REGISTRY)
    assert isinstance(build_reward_tracker(), RaceV2RewardTracker)


def test_race_v2_multiplies_time_penalty_by_frames_run() -> None:
    tracker = RaceV2RewardTracker(
        RaceV2RewardWeights(
            progress_scale=0.0,
            time_penalty_per_frame=-0.01,
        )
    )
    tracker.reset(_telemetry(race_distance=100.0))

    step = tracker.step_summary(
        _summary(max_race_distance=100.0, frames_run=3),
        _telemetry(race_distance=100.0),
    )

    assert step.reward == pytest.approx(-0.03)
    assert step.breakdown == {"time": -0.03}


def _telemetry(
    *,
    race_distance: float,
    state_flags: int = FLAG_ACTIVE,
    position: int = 30,
    energy: float = 178.0,
    boost_timer: int = 0,
    race_time_ms: int = 0,
    speed_kph: float = 100.0,
    ) -> FZeroXTelemetry:
    return FZeroXTelemetry(
        system_ram_size=0x00800000,
        game_frame_count=100,
        game_mode_raw=1,
        game_mode_name="gp_race",
        course_index=0,
        in_race_mode=True,
        player=PlayerTelemetry(
            state_flags=state_flags,
            state_labels=(),
            speed_raw=0.0,
            speed_kph=speed_kph,
            energy=energy,
            max_energy=178.0,
            boost_timer=boost_timer,
            race_distance=race_distance,
            laps_completed_distance=0.0,
            lap_distance=race_distance,
            race_distance_position=race_distance,
            race_time_ms=race_time_ms,
            lap=1,
            laps_completed=0,
            position=position,
            character=0,
            machine_index=0,
        ),
    )


def _summary(
    *,
    max_race_distance: float,
    frames_run: int = 1,
    reverse_progress_total: float = 0.0,
    energy_loss_total: float = 0.0,
    entered_state_flags: int = 0,
) -> StepSummary:
    return StepSummary(
        frames_run=frames_run,
        max_race_distance=max_race_distance,
        reverse_progress_total=reverse_progress_total,
        consecutive_reverse_frames=0,
        energy_loss_total=energy_loss_total,
        consecutive_low_speed_frames=0,
        entered_state_flags=entered_state_flags,
        final_frame_index=frames_run,
    )
