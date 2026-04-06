# tests/test_reward.py
from rl_fzerox.core.game import FZeroXTelemetry, PlayerTelemetry, RewardTracker


def test_reward_tracker_rewards_new_best_progress_only() -> None:
    tracker = RewardTracker()
    tracker.reset(_telemetry(race_distance=100.0))

    first = tracker.step(_telemetry(race_distance=130.0))
    second = tracker.step(_telemetry(race_distance=120.0))

    assert first.reward == 0.03
    assert first.terminated is False
    assert first.breakdown == {"progress": 0.03}
    assert second.reward == 0.0
    assert second.terminated is False


def test_reward_tracker_applies_terminal_penalties_and_finish_bonus_once() -> None:
    tracker = RewardTracker()
    tracker.reset(_telemetry(race_distance=0.0, state_flags=1 << 30))

    crashed = tracker.step(_telemetry(race_distance=10.0, state_flags=(1 << 30) | (1 << 27)))
    repeated = tracker.step(_telemetry(race_distance=10.0, state_flags=(1 << 30) | (1 << 27)))
    finished = RewardTracker()
    finished.reset(_telemetry(race_distance=0.0, state_flags=1 << 30))
    done = finished.step(_telemetry(race_distance=100.0, state_flags=(1 << 30) | (1 << 25)))

    assert crashed.terminated is True
    assert crashed.breakdown["progress"] == 0.01
    assert crashed.breakdown["crashed"] == -20.0
    assert crashed.reward == -19.99
    assert repeated.reward == 0.0
    assert repeated.terminated is True
    assert done.terminated is True
    assert done.breakdown["finished"] == 50.0
    assert done.reward == 50.1


def _telemetry(
    *,
    race_distance: float,
    state_flags: int = 1 << 30,
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
            speed_kph=0.0,
            energy=178.0,
            max_energy=178.0,
            boost_timer=0,
            race_distance=race_distance,
            laps_completed_distance=0.0,
            lap_distance=race_distance,
            race_distance_position=race_distance,
            race_time_ms=0,
            lap=1,
            laps_completed=0,
            position=30,
            character=0,
            machine_index=0,
        ),
    )
