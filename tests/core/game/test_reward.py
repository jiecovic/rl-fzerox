# tests/core/game/test_reward.py
import pytest

from rl_fzerox.core.game import (
    FZeroXTelemetry,
    PlayerTelemetry,
    RewardTracker,
    RewardWeights,
)
from rl_fzerox.core.game.flags import COURSE_EFFECT_PIT, FLAG_AIRBORNE


def test_reward_tracker_rewards_new_best_progress_only() -> None:
    tracker = RewardTracker()
    tracker.reset(_telemetry(race_distance=100.0))

    first = tracker.step(_telemetry(race_distance=130.0))
    second = tracker.step(_telemetry(race_distance=120.0))

    assert first.reward == 0.03
    assert first.terminated is False
    assert first.breakdown == {"progress": 0.03}
    assert second.reward == -0.01
    assert second.breakdown == {"reverse_progress": -0.01}
    assert second.terminated is False


def test_reward_tracker_ignores_small_progress_noise() -> None:
    tracker = RewardTracker()
    tracker.reset(_telemetry(race_distance=100.0))

    step = tracker.step(_telemetry(race_distance=100.2))

    assert step.reward == 0.0
    assert step.breakdown == {}


def test_reward_tracker_applies_terminal_penalties_and_finish_bonus_once() -> None:
    tracker = RewardTracker()
    tracker.reset(_telemetry(race_distance=0.0, state_flags=1 << 30))

    crashed = tracker.step(_telemetry(race_distance=10.0, state_flags=(1 << 30) | (1 << 27)))
    repeated = tracker.step(_telemetry(race_distance=10.0, state_flags=(1 << 30) | (1 << 27)))
    finished = RewardTracker()
    finished.reset(_telemetry(race_distance=0.0, state_flags=1 << 30))
    done = finished.step(
        _telemetry(
            race_distance=100.0,
            state_flags=(1 << 30) | (1 << 25),
            position=1,
        )
    )

    assert crashed.terminated is True
    assert crashed.breakdown["progress"] == 0.01
    assert crashed.breakdown["crashed"] == -20.0
    assert crashed.reward == -19.99
    assert repeated.reward == 0.0
    assert repeated.terminated is True
    assert done.terminated is True
    assert done.breakdown["finished"] == 100.0
    assert done.breakdown["finish_position"] == 116.0
    assert done.reward == 216.1


def test_reward_tracker_applies_stronger_collision_penalty_once() -> None:
    tracker = RewardTracker()
    tracker.reset(_telemetry(race_distance=100.0))

    step = tracker.step(_telemetry(race_distance=100.0, state_flags=(1 << 30) | (1 << 13)))
    repeated = tracker.step(
        _telemetry(race_distance=100.0, state_flags=(1 << 30) | (1 << 13))
    )

    assert step.terminated is False
    assert step.breakdown["collision_recoil"] == -4.0
    assert step.reward == -4.0
    assert repeated.reward == 0.0


def test_reward_tracker_rewards_dash_pad_boost_entry_once() -> None:
    tracker = RewardTracker()
    tracker.reset(_telemetry(race_distance=100.0))

    step = tracker.step(_telemetry(race_distance=100.0, state_flags=(1 << 30) | (1 << 24)))
    repeated = tracker.step(
        _telemetry(race_distance=100.0, state_flags=(1 << 30) | (1 << 24))
    )

    assert step.terminated is False
    assert step.breakdown == {"dash_pad_boost": 2.0}
    assert step.reward == 2.0
    assert repeated.reward == 0.0
    assert repeated.breakdown == {}


def test_reward_tracker_requires_progress_before_rewarding_dash_pad_again() -> None:
    tracker = RewardTracker(
        RewardWeights(
            dash_pad_boost_reward=0.75,
            dash_pad_min_progress=500.0,
        )
    )
    tracker.reset(_telemetry(race_distance=100.0))

    first = tracker.step(_telemetry(race_distance=120.0, state_flags=(1 << 30) | (1 << 24)))
    tracker.step(_telemetry(race_distance=120.0, state_flags=1 << 30))
    blocked = tracker.step(_telemetry(race_distance=540.0, state_flags=(1 << 30) | (1 << 24)))
    tracker.step(_telemetry(race_distance=540.0, state_flags=1 << 30))
    allowed = tracker.step(_telemetry(race_distance=640.0, state_flags=(1 << 30) | (1 << 24)))

    assert first.reward == 0.77
    assert first.breakdown == {"progress": 0.02, "dash_pad_boost": 0.75}
    assert blocked.reward == 0.42
    assert blocked.breakdown == {"progress": 0.42}
    assert allowed.reward == 0.85
    assert allowed.breakdown == {"progress": 0.1, "dash_pad_boost": 0.75}


def test_reward_tracker_rewards_each_checkpoint_only_once() -> None:
    tracker = RewardTracker(
        RewardWeights(
            progress_scale=0.0,
            checkpoint_spacing=3_000.0,
            checkpoint_fast_time_ms=3_000,
            checkpoint_slow_time_ms=8_000,
            checkpoint_fast_bonus=1.0,
            checkpoint_slow_bonus=0.25,
        )
    )
    tracker.reset(_telemetry(race_distance=0.0, race_time_ms=0))

    first = tracker.step(_telemetry(race_distance=3_100.0, race_time_ms=2_000))
    tracker.step(_telemetry(race_distance=2_800.0, race_time_ms=2_500))
    repeated = tracker.step(_telemetry(race_distance=3_200.0, race_time_ms=3_000))

    assert first.reward == 1.0
    assert first.breakdown == {"checkpoint": 1.0}
    assert repeated.reward == 0.0
    assert repeated.breakdown == {}


def test_reward_tracker_scales_checkpoint_bonus_with_race_time() -> None:
    fast_tracker = RewardTracker(
        RewardWeights(
            progress_scale=0.0,
            checkpoint_spacing=3_000.0,
            checkpoint_fast_time_ms=1_000,
            checkpoint_slow_time_ms=5_000,
            checkpoint_fast_bonus=1.0,
            checkpoint_slow_bonus=0.25,
        )
    )
    slow_tracker = RewardTracker(
        RewardWeights(
            progress_scale=0.0,
            checkpoint_spacing=3_000.0,
            checkpoint_fast_time_ms=1_000,
            checkpoint_slow_time_ms=5_000,
            checkpoint_fast_bonus=1.0,
            checkpoint_slow_bonus=0.25,
        )
    )
    fast_tracker.reset(_telemetry(race_distance=0.0, race_time_ms=0))
    slow_tracker.reset(_telemetry(race_distance=0.0, race_time_ms=0))

    fast = fast_tracker.step(_telemetry(race_distance=3_100.0, race_time_ms=1_000))
    slow = slow_tracker.step(_telemetry(race_distance=3_100.0, race_time_ms=5_000))

    assert fast.reward == 1.0
    assert fast.breakdown == {"checkpoint": 1.0}
    assert slow.reward == 0.25
    assert slow.breakdown == {"checkpoint": 0.25}


def test_reward_tracker_caps_checkpoint_reward_on_large_single_step_jump() -> None:
    tracker = RewardTracker(
        RewardWeights(
            progress_scale=0.0,
            checkpoint_spacing=3_000.0,
            checkpoint_fast_time_ms=3_000,
            checkpoint_slow_time_ms=8_000,
            checkpoint_fast_bonus=1.0,
            checkpoint_slow_bonus=0.25,
        )
    )
    tracker.reset(_telemetry(race_distance=100.0, race_time_ms=0))

    step = tracker.step(_telemetry(race_distance=9_100.0, race_time_ms=2_000))

    assert step.reward == 1.0
    assert step.breakdown == {"checkpoint": 1.0}


def test_reward_tracker_scales_low_speed_penalty_by_speed_deficit() -> None:
    tracker = RewardTracker(
        RewardWeights(
            progress_scale=0.0,
            low_speed_threshold_kph=100.0,
            low_speed_penalty=-0.2,
        )
    )
    tracker.reset(_telemetry(race_distance=100.0, speed_kph=50.0))

    step = tracker.step(_telemetry(race_distance=100.0, speed_kph=50.0))

    assert step.reward == -0.1
    assert step.breakdown == {"low_speed": -0.1}


def test_reward_tracker_applies_max_low_speed_penalty_at_standstill() -> None:
    tracker = RewardTracker(
        RewardWeights(
            progress_scale=0.0,
            low_speed_threshold_kph=100.0,
            low_speed_penalty=-0.2,
        )
    )
    tracker.reset(_telemetry(race_distance=100.0, speed_kph=0.0))

    step = tracker.step(_telemetry(race_distance=100.0, speed_kph=0.0))

    assert step.reward == -0.2
    assert step.breakdown == {"low_speed": -0.2}


def test_reward_tracker_skips_low_speed_penalty_while_airborne() -> None:
    tracker = RewardTracker(
        RewardWeights(
            progress_scale=0.0,
            low_speed_threshold_kph=100.0,
            low_speed_penalty=-0.2,
        )
    )
    tracker.reset(_telemetry(race_distance=100.0, speed_kph=20.0))

    step = tracker.step(
        _telemetry(
            race_distance=100.0,
            speed_kph=20.0,
            state_flags=(1 << 30) | FLAG_AIRBORNE,
        )
    )

    assert step.reward == 0.0
    assert step.breakdown == {}


def test_reward_tracker_skips_low_speed_penalty_at_or_above_threshold() -> None:
    tracker = RewardTracker(
        RewardWeights(
            progress_scale=0.0,
            low_speed_threshold_kph=100.0,
            low_speed_penalty=-0.2,
        )
    )
    tracker.reset(_telemetry(race_distance=100.0, speed_kph=100.0))

    threshold = tracker.step(_telemetry(race_distance=100.0, speed_kph=100.0))
    above = tracker.step(_telemetry(race_distance=100.0, speed_kph=120.0))

    assert threshold.reward == 0.0
    assert threshold.breakdown == {}
    assert above.reward == 0.0
    assert above.breakdown == {}


def test_reward_tracker_penalizes_stalling_after_grace_period() -> None:
    tracker = RewardTracker(
        RewardWeights(
            progress_epsilon=0.5,
            stall_grace_steps=2,
            stall_penalty=-0.25,
        )
    )
    tracker.reset(_telemetry(race_distance=100.0))

    first = tracker.step(_telemetry(race_distance=100.0))
    second = tracker.step(_telemetry(race_distance=100.0))
    third = tracker.step(_telemetry(race_distance=100.0))

    assert first.reward == 0.0
    assert second.reward == 0.0
    assert third.reward == -0.25
    assert third.breakdown == {"stall": -0.25}


def test_reward_tracker_penalizes_reverse_driving_and_stalling_together() -> None:
    tracker = RewardTracker(
        RewardWeights(
            progress_scale=0.0,
            reverse_progress_scale=0.001,
            progress_epsilon=0.5,
            stall_grace_steps=0,
            stall_penalty=-0.25,
        )
    )
    tracker.reset(_telemetry(race_distance=100.0))

    step = tracker.step(_telemetry(race_distance=90.0))

    assert step.reward == -0.26
    assert step.breakdown == {
        "reverse_progress": -0.01,
        "stall": -0.25,
    }


def test_reward_tracker_resets_stall_counter_on_progress() -> None:
    tracker = RewardTracker(
        RewardWeights(
            progress_epsilon=0.5,
            stall_grace_steps=1,
            stall_penalty=-0.25,
        )
    )
    tracker.reset(_telemetry(race_distance=100.0))

    stalled = tracker.step(_telemetry(race_distance=100.0))
    recovered = tracker.step(_telemetry(race_distance=110.0))
    stalled_again = tracker.step(_telemetry(race_distance=110.0))

    assert stalled.reward == 0.0
    assert recovered.reward == 0.01
    assert recovered.breakdown == {"progress": 0.01}
    assert stalled_again.reward == 0.0


def test_reward_tracker_returns_stuck_truncation_penalty() -> None:
    tracker = RewardTracker(
        RewardWeights(stuck_truncation_penalty=-7.5)
    )

    penalty, label = tracker.truncation_penalty("stuck")
    none_penalty, none_label = tracker.truncation_penalty("timeout")

    assert penalty == -7.5
    assert label == "stuck_truncation"
    assert none_penalty == 0.0
    assert none_label is None


def test_reward_tracker_returns_wrong_way_truncation_penalty() -> None:
    tracker = RewardTracker(
        RewardWeights(wrong_way_truncation_penalty=-12.0)
    )

    penalty, label = tracker.truncation_penalty("wrong_way")

    assert penalty == -12.0
    assert label == "wrong_way_truncation"


def test_reward_tracker_penalizes_energy_loss_but_not_energy_gain() -> None:
    tracker = RewardTracker(
        RewardWeights(
            energy_loss_epsilon=0.1,
            energy_loss_penalty_scale=0.1,
        )
    )
    tracker.reset(_telemetry(race_distance=100.0, energy=178.0))

    loss = tracker.step(_telemetry(race_distance=100.0, energy=174.0))
    gain = tracker.step(_telemetry(race_distance=100.0, energy=176.0))

    assert loss.reward == -0.4
    assert loss.breakdown == {"energy_loss": -0.4}
    assert gain.reward == 0.0
    assert gain.breakdown == {}


def test_reward_tracker_penalizes_manual_boost_start_at_low_energy() -> None:
    tracker = RewardTracker(
        RewardWeights(
            progress_scale=0.0,
            low_energy_boost_threshold_ratio=0.1,
            low_energy_boost_penalty=-6.0,
        )
    )
    tracker.reset(_telemetry(race_distance=100.0, energy=15.0, boost_timer=0))

    step = tracker.step(_telemetry(race_distance=100.0, energy=14.0, boost_timer=100))

    assert step.reward == -6.1
    assert step.breakdown == {
        "energy_loss": -0.1,
        "low_energy_boost": -6.0,
    }


def test_reward_tracker_scales_manual_boost_penalty_with_remaining_energy() -> None:
    tracker = RewardTracker(
        RewardWeights(
            progress_scale=0.0,
            low_energy_boost_threshold_ratio=0.1,
            low_energy_boost_penalty=-6.0,
        )
    )
    tracker.reset(_telemetry(race_distance=100.0, energy=89.0, boost_timer=0))

    step = tracker.step(_telemetry(race_distance=100.0, energy=87.0, boost_timer=100))

    assert step.reward == pytest.approx(-3.5333333333333337)
    assert step.breakdown == {
        "energy_loss": -0.2,
        "low_energy_boost": pytest.approx(-3.3333333333333335),
    }


def test_reward_tracker_skips_manual_boost_penalty_at_full_energy() -> None:
    tracker = RewardTracker(
        RewardWeights(
            progress_scale=0.0,
            low_energy_boost_threshold_ratio=0.1,
            low_energy_boost_penalty=-6.0,
        )
    )
    tracker.reset(_telemetry(race_distance=100.0, energy=178.0, boost_timer=0))

    step = tracker.step(_telemetry(race_distance=100.0, energy=176.0, boost_timer=100))

    assert step.reward == -0.2
    assert step.breakdown == {"energy_loss": -0.2}


def test_reward_tracker_skips_low_energy_boost_penalty_for_dash_pad() -> None:
    tracker = RewardTracker(
        RewardWeights(
            progress_scale=0.0,
            dash_pad_boost_reward=2.0,
            low_energy_boost_threshold_ratio=0.1,
            low_energy_boost_penalty=-6.0,
        )
    )
    tracker.reset(_telemetry(race_distance=100.0, energy=15.0, boost_timer=0))

    step = tracker.step(
        _telemetry(
            race_distance=650.0,
            energy=14.0,
            boost_timer=100,
            state_flags=(1 << 30) | (1 << 24),
        )
    )

    assert step.reward == 1.9
    assert step.breakdown == {
        "energy_loss": -0.1,
        "dash_pad_boost": 2.0,
    }


def test_reward_tracker_rewards_forward_refill_while_on_strip() -> None:
    tracker = RewardTracker(
        RewardWeights(
            progress_scale=0.0,
            refill_reward_energy_cap=20.0,
            refill_reward_scale=0.05,
        )
    )
    tracker.reset(_telemetry(race_distance=100.0, energy=20.0))

    enter = tracker.step(
        _telemetry(
            race_distance=150.0,
            energy=20.0,
            state_flags=(1 << 30) | COURSE_EFFECT_PIT,
        )
    )
    on_strip = tracker.step(
        _telemetry(
            race_distance=520.0,
            energy=30.0,
            state_flags=(1 << 30) | COURSE_EFFECT_PIT,
        )
    )
    exit_step = tracker.step(_telemetry(race_distance=720.0, energy=35.0))

    assert enter.reward == 0.0
    assert enter.breakdown == {}
    assert on_strip.reward == pytest.approx(0.4438202247191011)
    assert on_strip.breakdown == {"refill": pytest.approx(0.4438202247191011)}
    assert exit_step.reward == 0.0
    assert exit_step.breakdown == {}


def test_reward_tracker_scales_refill_reward_by_missing_energy_at_entry() -> None:
    tracker = RewardTracker(
        RewardWeights(
            progress_scale=0.0,
            refill_reward_energy_cap=20.0,
            refill_reward_scale=0.05,
        )
    )
    tracker.reset(_telemetry(race_distance=100.0, energy=120.0))

    tracker.step(
        _telemetry(
            race_distance=150.0,
            energy=120.0,
            state_flags=(1 << 30) | COURSE_EFFECT_PIT,
        )
    )
    tracker.step(
        _telemetry(
            race_distance=520.0,
            energy=130.0,
            state_flags=(1 << 30) | COURSE_EFFECT_PIT,
        )
    )
    refill_step = tracker.step(
        _telemetry(
            race_distance=720.0,
            energy=135.0,
            state_flags=(1 << 30) | COURSE_EFFECT_PIT,
        )
    )

    assert refill_step.reward == pytest.approx(0.08146067415730338)
    assert refill_step.breakdown == {"refill": pytest.approx(0.08146067415730338)}


def test_reward_tracker_skips_refill_reward_without_forward_progress() -> None:
    tracker = RewardTracker(
        RewardWeights(
            progress_scale=0.0,
            reverse_progress_scale=0.0,
            refill_reward_energy_cap=20.0,
            refill_reward_scale=0.05,
        )
    )
    tracker.reset(_telemetry(race_distance=1_000.0, energy=20.0))

    tracker.step(
        _telemetry(
            race_distance=900.0,
            energy=20.0,
            state_flags=(1 << 30) | COURSE_EFFECT_PIT,
        )
    )
    reverse_refill = tracker.step(
        _telemetry(
            race_distance=880.0,
            energy=30.0,
            state_flags=(1 << 30) | COURSE_EFFECT_PIT,
        )
    )
    exit_step = tracker.step(_telemetry(race_distance=860.0, energy=35.0))

    assert reverse_refill.reward == 0.0
    assert reverse_refill.breakdown == {}
    assert exit_step.reward == 0.0
    assert exit_step.breakdown == {}


def _telemetry(
    *,
    race_distance: float,
    state_flags: int = 1 << 30,
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
