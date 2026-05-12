# tests/ui/test_live_episode.py
from __future__ import annotations

from dataclasses import dataclass

from rl_fzerox.ui.watch.view.live_episode import EpisodeLiveSeriesTracker


@dataclass(frozen=True)
class _Snapshot:
    episode: int
    policy_decision_frame: bool
    info: dict[str, object]
    episode_reward: float


def test_live_episode_tracker_uses_decision_frames_only() -> None:
    tracker = EpisodeLiveSeriesTracker()

    tracker.observe_snapshot(
        _Snapshot(
            episode=3,
            policy_decision_frame=False,
            info={
                "episode_step": 5,
                "episode_completion_fraction": 0.2,
                "speed_kph": 300.0,
            },
            episode_reward=1.0,
        ),
        action_repeat=2,
    )
    tracker.observe_snapshot(
        _Snapshot(
            episode=3,
            policy_decision_frame=True,
            info={
                "episode_step": 6,
                "episode_completion_fraction": 0.25,
                "speed_kph": 420.0,
            },
            episode_reward=1.5,
        ),
        action_repeat=2,
    )

    snapshot = tracker.snapshot()
    assert snapshot is not None
    assert snapshot.env_steps == (3,)
    assert snapshot.speed_kph == (420.0,)
    assert snapshot.returns == (1.5,)
    assert snapshot.current_return == 1.5
    assert snapshot.current_progress == 0.25
    assert snapshot.max_progress == 0.25


def test_live_episode_tracker_resets_on_new_episode() -> None:
    tracker = EpisodeLiveSeriesTracker()

    tracker.observe_snapshot(
        _Snapshot(
            episode=1,
            policy_decision_frame=True,
            info={
                "episode_step": 4,
                "episode_completion_fraction": 0.1,
                "speed_kph": 250.0,
            },
            episode_reward=0.5,
        ),
        action_repeat=2,
    )
    tracker.observe_snapshot(
        _Snapshot(
            episode=1,
            policy_decision_frame=True,
            info={
                "episode_step": 6,
                "episode_completion_fraction": 0.2,
                "speed_kph": 320.0,
            },
            episode_reward=0.25,
        ),
        action_repeat=2,
    )
    tracker.observe_snapshot(
        _Snapshot(
            episode=2,
            policy_decision_frame=True,
            info={
                "episode_step": 2,
                "episode_completion_fraction": 0.05,
                "speed_kph": 180.0,
            },
            episode_reward=0.1,
        ),
        action_repeat=2,
    )

    snapshot = tracker.snapshot()
    assert snapshot is not None
    assert snapshot.episode == 2
    assert snapshot.env_steps == (1,)
    assert snapshot.speed_kph == (180.0,)
    assert snapshot.returns == (0.1,)
    assert snapshot.current_return == 0.1
    assert snapshot.current_progress == 0.05
    assert snapshot.max_progress == 0.05
