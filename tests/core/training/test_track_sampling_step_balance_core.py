# tests/core/training/test_track_sampling_step_balance_core.py

import pytest

from rl_fzerox.core.training.session.callbacks.track_sampling import (
    StepBalancedTrackSamplingController,
)
from tests.core.training.track_sampling_support import resolved_track_sampling_courses


def test_step_balance_controller_raises_short_track_weight() -> None:
    controller = StepBalancedTrackSamplingController(
        resolved_courses=resolved_track_sampling_courses({"mute": 1.0, "silence": 1.0}),
        action_repeat=2,
        update_episodes=2,
        ema_alpha=1.0,
        max_weight_scale=5.0,
    )

    weights = controller.record_episodes(
        (
            {"track_id": "mute", "episode_step": 100},
            {"track_id": "silence", "episode_step": 400},
        )
    )

    assert weights is not None
    assert weights == pytest.approx({"mute": 1.92, "silence": 0.08})


def test_step_balance_controller_can_use_monitor_length_fallback() -> None:
    controller = StepBalancedTrackSamplingController(
        resolved_courses=resolved_track_sampling_courses({"mute": 1.0, "silence": 1.0}),
        action_repeat=3,
        update_episodes=2,
        ema_alpha=1.0,
        max_weight_scale=5.0,
    )

    weights = controller.record_episodes(
        (
            {"track_id": "mute", "l": 10},
            {"track_id": "silence", "l": 40},
        )
    )

    assert weights is not None
    assert weights == pytest.approx({"mute": 1.92, "silence": 0.08})


def test_step_balance_controller_skips_tensorboard_track_logs_by_default() -> None:
    controller = StepBalancedTrackSamplingController(
        resolved_courses=resolved_track_sampling_courses({"mute": 1.0, "silence": 1.0}),
        action_repeat=1,
        update_episodes=2,
        ema_alpha=1.0,
        max_weight_scale=5.0,
    )

    controller.record_episodes(
        (
            {"track_id": "mute", "episode_step": 100},
            {"track_id": "silence", "episode_step": 300},
        )
    )
    values = controller.log_values()

    assert values == {}


def test_step_balance_controller_keeps_tensorboard_logs_compact() -> None:
    controller = StepBalancedTrackSamplingController(
        resolved_courses=resolved_track_sampling_courses({"mute": 1.0, "silence": 1.0}),
        action_repeat=1,
        update_episodes=2,
        ema_alpha=1.0,
        max_weight_scale=5.0,
        log_details=True,
    )

    controller.record_episodes(
        (
            {"track_id": "mute", "episode_step": 100},
            {"track_id": "silence", "episode_step": 300},
        )
    )
    values = controller.log_values()

    assert values == {
        "track_sampling/course_count": 2.0,
        "track_sampling/update_count": 1.0,
    }
    assert not any(key.startswith("track_sampling/mute/") for key in values)
    assert not any(key.startswith("track_sampling/silence/") for key in values)


def test_step_balance_controller_tracks_finished_episode_counts() -> None:
    controller = StepBalancedTrackSamplingController(
        resolved_courses=resolved_track_sampling_courses({"mute": 1.0, "silence": 1.0}),
        action_repeat=2,
        update_episodes=2,
        ema_alpha=1.0,
        max_weight_scale=5.0,
    )

    controller.record_episodes(
        (
            {"track_id": "mute", "episode_step": 100, "termination_reason": "finished"},
            {"track_id": "silence", "episode_step": 400, "termination_reason": "stalled"},
        )
    )

    runtime = controller.runtime_state()

    assert {entry.track_id: entry.finished_episode_count for entry in runtime.entries} == {
        "mute": 1,
        "silence": 0,
    }


def test_step_balance_controller_ignores_alt_baseline_episodes_for_stats() -> None:
    controller = StepBalancedTrackSamplingController(
        resolved_courses=resolved_track_sampling_courses({"mute": 1.0, "silence": 1.0}),
        action_repeat=1,
        update_episodes=1,
        ema_alpha=1.0,
        max_weight_scale=5.0,
    )

    weights = controller.record_episodes(
        (
            {
                "track_id": "mute",
                "track_alt_baseline_id": "alt-a",
                "episode_step": 12,
                "episode_completion_fraction": 1.0,
                "termination_reason": "finished",
            },
        )
    )

    runtime = controller.runtime_state()
    assert weights is None
    assert {entry.track_id: entry.completed_frames for entry in runtime.entries} == {
        "mute": 0,
        "silence": 0,
    }
    assert {entry.track_id: entry.episode_count for entry in runtime.entries} == {
        "mute": 0,
        "silence": 0,
    }
