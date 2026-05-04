# tests/core/training/test_track_sampling_balance.py
from pathlib import Path

import pytest

from rl_fzerox.core.config.schema import (
    CurriculumConfig,
    EnvConfig,
    TrackSamplingConfig,
    TrackSamplingEntryConfig,
)
from rl_fzerox.core.training.session.callbacks.track_sampling import (
    StepBalancedTrackSamplingController,
    load_track_sampling_runtime_state,
    save_track_sampling_runtime_state,
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
    assert weights["mute"] > weights["silence"]
    assert weights["mute"] + weights["silence"] == pytest.approx(2.0)


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
    assert weights["mute"] > weights["silence"]


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

    assert values["track_sampling/mute/prob"] + values["track_sampling/silence/prob"] == (
        pytest.approx(1.0)
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


def test_step_balance_controller_builds_from_env_config() -> None:
    controller = StepBalancedTrackSamplingController.from_configs(
        env_config=EnvConfig(
            action_repeat=2,
            track_sampling=TrackSamplingConfig(
                enabled=True,
                sampling_mode="step_balanced",
                entries=(
                    TrackSamplingEntryConfig(id="mute", weight=1.0),
                    TrackSamplingEntryConfig(id="silence", weight=1.0),
                ),
                step_balance_update_episodes=3,
            ),
        ),
        curriculum_config=CurriculumConfig(),
    )

    assert controller is not None
    assert controller.record_episodes(({"track_id": "mute", "episode_step": 100},)) is None


def test_step_balance_controller_logs_unique_courses_by_course_id() -> None:
    controller = StepBalancedTrackSamplingController.from_configs(
        env_config=EnvConfig(
            track_sampling=TrackSamplingConfig(
                enabled=True,
                sampling_mode="step_balanced",
                entries=(
                    TrackSamplingEntryConfig(
                        id="big_blue_time_attack_blue_falcon_balanced",
                        course_id="big_blue",
                        weight=1.0,
                    ),
                    TrackSamplingEntryConfig(
                        id="silence_time_attack_blue_falcon_balanced",
                        course_id="silence",
                        weight=1.0,
                    ),
                ),
                step_balance_log_details=True,
            ),
        ),
        curriculum_config=CurriculumConfig(),
    )

    assert controller is not None
    values = controller.log_values()

    assert "track_sampling/big_blue/prob" in values
    assert "track_sampling/silence/prob" in values
    assert "track_sampling/big_blue_time_attack_blue_falcon_balanced/prob" not in values


def test_step_balance_controller_aggregates_duplicate_courses() -> None:
    controller = StepBalancedTrackSamplingController.from_configs(
        env_config=EnvConfig(
            track_sampling=TrackSamplingConfig(
                enabled=True,
                sampling_mode="step_balanced",
                entries=(
                    TrackSamplingEntryConfig(
                        id="big_blue_time_attack_blue_falcon_balanced",
                        course_id="big_blue",
                        weight=1.0,
                    ),
                    TrackSamplingEntryConfig(
                        id="big_blue_gp_blue_falcon_balanced",
                        course_id="big_blue",
                        weight=1.0,
                    ),
                ),
                step_balance_log_details=True,
            ),
        ),
        curriculum_config=CurriculumConfig(),
    )

    assert controller is None


def test_step_balance_controller_aggregates_duplicate_course_entries() -> None:
    controller = StepBalancedTrackSamplingController(
        track_base_weights={
            "mute_blue_falcon": 1.0,
            "mute_white_cat": 1.0,
            "silence_blue_falcon": 1.0,
        },
        action_repeat=1,
        update_episodes=2,
        ema_alpha=1.0,
        max_weight_scale=5.0,
        track_course_keys={
            "mute_blue_falcon": "mute_city",
            "mute_white_cat": "mute_city",
            "silence_blue_falcon": "silence",
        },
        track_log_keys={
            "mute_blue_falcon": "mute_city",
            "mute_white_cat": "mute_city",
            "silence_blue_falcon": "silence",
        },
        track_labels={
            "mute_blue_falcon": "Mute City",
            "mute_white_cat": "Mute City",
            "silence_blue_falcon": "Silence",
        },
    )

    weights = controller.record_episodes(
        (
            {"track_id": "mute_blue_falcon", "episode_step": 100},
            {"track_id": "silence_blue_falcon", "episode_step": 400},
        )
    )

    assert weights is not None
    assert weights["mute_blue_falcon"] == pytest.approx(weights["mute_white_cat"])
    assert weights["mute_blue_falcon"] + weights["mute_white_cat"] > weights["silence_blue_falcon"]
    runtime = controller.runtime_state()
    assert {entry.track_id: entry.completed_frames for entry in runtime.entries} == {
        "mute_city": 100,
        "silence": 400,
    }
    assert {entry.track_id: entry.episode_count for entry in runtime.entries} == {
        "mute_city": 1,
        "silence": 1,
    }


def test_step_balance_controller_state_round_trip_and_restore(tmp_path: Path) -> None:
    controller = StepBalancedTrackSamplingController(
        track_base_weights={"mute": 1.0, "silence": 1.0},
        action_repeat=2,
        update_episodes=2,
        ema_alpha=0.5,
        max_weight_scale=5.0,
        track_log_keys={"mute": "mute_city", "silence": "silence"},
        track_labels={"mute": "Mute City", "silence": "Silence"},
    )
    controller.record_episodes(
        (
            {"track_id": "mute", "episode_step": 100, "termination_reason": "finished"},
            {"track_id": "silence", "episode_step": 400, "termination_reason": "stalled"},
        )
    )
    state_path = tmp_path / "track_sampling_state.json"
    save_track_sampling_runtime_state(state_path, controller.runtime_state())

    restored_state = load_track_sampling_runtime_state(state_path)

    assert restored_state is not None
    restored = StepBalancedTrackSamplingController(
        track_base_weights={"mute": 1.0, "silence": 1.0},
        action_repeat=2,
        update_episodes=2,
        ema_alpha=0.5,
        max_weight_scale=5.0,
        track_log_keys={"mute": "mute_city", "silence": "silence"},
        track_labels={"mute": "Mute City", "silence": "Silence"},
        restored_state=restored_state,
    )

    assert restored.current_weights() == pytest.approx(controller.current_weights())
    restored_runtime = restored.runtime_state()
    assert {entry.track_id: entry.completed_frames for entry in restored_runtime.entries} == {
        "mute_city": 100,
        "silence": 400,
    }
    assert {entry.track_id: entry.episode_count for entry in restored_runtime.entries} == {
        "mute_city": 1,
        "silence": 1,
    }
    assert {entry.track_id: entry.finished_episode_count for entry in restored_runtime.entries} == {
        "mute_city": 1,
        "silence": 0,
    }
    assert {entry.track_id: entry.success_sample_count for entry in restored_runtime.entries} == {
        "mute_city": 1,
        "silence": 1,
    }
