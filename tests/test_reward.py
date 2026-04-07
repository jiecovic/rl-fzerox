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
    assert done.breakdown["finish_position"] == 58.0
    assert done.reward == 158.1


def test_reward_tracker_applies_stronger_collision_penalty_once() -> None:
    tracker = RewardTracker()
    tracker.reset(_telemetry(race_distance=100.0))

    step = tracker.step(_telemetry(race_distance=100.0, state_flags=(1 << 30) | (1 << 13)))
    repeated = tracker.step(
        _telemetry(race_distance=100.0, state_flags=(1 << 30) | (1 << 13))
    )

    assert step.terminated is False
    assert step.breakdown["collision_recoil"] == -1.0
    assert step.reward == -1.0
    assert repeated.reward == 0.0


def _telemetry(
    *,
    race_distance: float,
    state_flags: int = 1 << 30,
    position: int = 30,
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
            position=position,
            character=0,
            machine_index=0,
        ),
    )
