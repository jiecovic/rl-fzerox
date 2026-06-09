# tests/core/training/test_track_sampling_step_balance_core.py

import pytest

from rl_fzerox.core.training.session.callbacks.track_sampling import (
    StepBalancedTrackSamplingController,
)


def test_step_balance_controller_raises_short_track_weight() -> None:
    controller = StepBalancedTrackSamplingController(
        track_base_weights={"mute": 1.0, "silence": 1.0},
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
        track_base_weights={"mute": 1.0, "silence": 1.0},
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
        track_base_weights={"mute": 1.0, "silence": 1.0},
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


def test_step_balance_controller_can_log_detailed_distribution_shares() -> None:
    controller = StepBalancedTrackSamplingController(
        track_base_weights={"mute": 1.0, "silence": 1.0},
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

    assert values["track_sampling/mute/prob"] == pytest.approx(0.95)
    assert values["track_sampling/silence/prob"] == pytest.approx(0.05)
    assert values["track_sampling/mute/prob"] + values["track_sampling/silence/prob"] == (
        pytest.approx(1.0)
    )
    assert values["track_sampling/mute/expected_frame_share"] + values[
        "track_sampling/silence/expected_frame_share"
    ] == pytest.approx(1.0)
    assert values["track_sampling/mute/target_frame_share"] == pytest.approx(
        values["track_sampling/silence/target_frame_share"]
    )
    assert "track_sampling/mute/frame_share" not in values
    assert "track_sampling/silence/ema_episode_frames" not in values


def test_step_balance_controller_tracks_finished_episode_counts() -> None:
    controller = StepBalancedTrackSamplingController(
        track_base_weights={"mute": 1.0, "silence": 1.0},
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
