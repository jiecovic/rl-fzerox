# tests/core/training/test_track_sampling_step_balance_state.py
from pathlib import Path

import pytest

from rl_fzerox.core.domain.x_cup import X_CUP_COURSE
from rl_fzerox.core.runtime_spec.schema import (
    EnvConfig,
    TrackSamplingConfig,
    TrackSamplingEntryConfig,
)
from rl_fzerox.core.training.session.callbacks.track_sampling import (
    StepBalancedTrackSamplingController,
    TrackSamplingRuntimeEntry,
    TrackSamplingRuntimeState,
    load_track_sampling_runtime_state,
    save_track_sampling_runtime_state,
)
from tests.core.training.track_sampling_support import resolved_track_sampling_courses


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
    )

    assert controller is not None
    assert controller.record_episodes(({"track_id": "mute", "episode_step": 100},)) is None


def test_step_balance_controller_compacts_tensorboard_course_logs() -> None:
    controller = StepBalancedTrackSamplingController.from_configs(
        env_config=EnvConfig(
            track_sampling=TrackSamplingConfig(
                enabled=True,
                sampling_mode="step_balanced",
                entries=(
                    TrackSamplingEntryConfig(
                        id="big_blue",
                        course_id="big_blue",
                        weight=1.0,
                    ),
                    TrackSamplingEntryConfig(
                        id="silence",
                        course_id="silence",
                        weight=1.0,
                    ),
                ),
                step_balance_log_details=True,
            ),
        ),
    )

    assert controller is not None
    values = controller.log_values()

    assert values == {}
    assert "track_sampling/big_blue/prob" not in values
    assert "track_sampling/silence/prob" not in values
    assert "track_sampling/big_blue_time_attack_blue_falcon_balanced/prob" not in values


def test_step_balance_controller_can_suppress_generated_course_logs() -> None:
    controller = StepBalancedTrackSamplingController.from_configs(
        env_config=EnvConfig(
            track_sampling=TrackSamplingConfig(
                enabled=True,
                sampling_mode="step_balanced",
                entries=(
                    TrackSamplingEntryConfig(
                        id="x_cup_a",
                        course_id="x_cup_a",
                        course_index=X_CUP_COURSE.course_index,
                        mode=X_CUP_COURSE.race_mode,
                        generated_course_kind=X_CUP_COURSE.generated_kind,
                        generated_course_seed=1,
                        generated_course_hash="a",
                        log_per_course=False,
                        weight=1.0,
                    ),
                    TrackSamplingEntryConfig(
                        id="x_cup_b",
                        course_id="x_cup_b",
                        course_index=X_CUP_COURSE.course_index,
                        mode=X_CUP_COURSE.race_mode,
                        generated_course_kind=X_CUP_COURSE.generated_kind,
                        generated_course_seed=2,
                        generated_course_hash="b",
                        log_per_course=False,
                        weight=1.0,
                    ),
                ),
                step_balance_log_details=True,
                step_balance_update_episodes=2,
            ),
        ),
    )

    assert controller is not None
    weights = controller.record_episodes(
        (
            {"track_id": "x_cup_a", "episode_step": 100},
            {"track_id": "x_cup_b", "episode_step": 300},
        )
    )

    assert weights is not None
    assert set(weights) == {"x_cup_a", "x_cup_b"}
    values = controller.log_values()
    assert values == {}
    assert not any(key.startswith("track_sampling/x_cup_") for key in values)


def test_step_balance_controller_aggregates_duplicate_courses() -> None:
    controller = StepBalancedTrackSamplingController.from_configs(
        env_config=EnvConfig(
            track_sampling=TrackSamplingConfig(
                enabled=True,
                sampling_mode="step_balanced",
                entries=(
                    TrackSamplingEntryConfig(
                        id="big_blue",
                        course_id="big_blue",
                        weight=1.0,
                    ),
                    TrackSamplingEntryConfig(
                        id="big_blue",
                        course_id="big_blue",
                        weight=1.0,
                    ),
                ),
                step_balance_log_details=True,
            ),
        ),
    )

    assert controller is None


def test_step_balance_controller_aggregates_duplicate_course_entries() -> None:
    controller = StepBalancedTrackSamplingController(
        resolved_courses=resolved_track_sampling_courses(
            {
                "mute_blue_falcon": 1.0,
                "mute_white_cat": 1.0,
                "silence_blue_falcon": 1.0,
            },
            course_keys={
                "mute_blue_falcon": "mute_city",
                "mute_white_cat": "mute_city",
                "silence_blue_falcon": "silence",
            },
            log_keys={
                "mute_blue_falcon": "mute_city",
                "mute_white_cat": "mute_city",
                "silence_blue_falcon": "silence",
            },
            labels={
                "mute_blue_falcon": "Mute City",
                "mute_white_cat": "Mute City",
                "silence_blue_falcon": "Silence",
            },
        ),
        action_repeat=1,
        update_episodes=2,
        ema_alpha=1.0,
        max_weight_scale=5.0,
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
        resolved_courses=resolved_track_sampling_courses(
            {"mute": 1.0, "silence": 1.0},
            course_keys={"mute": "mute_city", "silence": "silence"},
            log_keys={"mute": "mute_city", "silence": "silence"},
            labels={"mute": "Mute City", "silence": "Silence"},
        ),
        action_repeat=2,
        update_episodes=2,
        ema_alpha=0.5,
        max_weight_scale=5.0,
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
    assert restored_state.sampling_mode == "step_balanced"
    restored = StepBalancedTrackSamplingController(
        resolved_courses=resolved_track_sampling_courses(
            {"mute": 1.0, "silence": 1.0},
            course_keys={"mute": "mute_city", "silence": "silence"},
            log_keys={"mute": "mute_city", "silence": "silence"},
            labels={"mute": "Mute City", "silence": "Silence"},
        ),
        action_repeat=2,
        update_episodes=2,
        ema_alpha=0.5,
        max_weight_scale=5.0,
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


def test_step_balance_controller_recomputes_restored_weights_from_ema_stats() -> None:
    restored = TrackSamplingRuntimeState(
        sampling_mode="step_balanced",
        action_repeat=2,
        update_episodes=2,
        ema_alpha=1.0,
        max_weight_scale=5.0,
        adaptive_completion_weight=0.35,
        adaptive_target_completion=0.9,
        adaptive_min_confidence_episodes=24,
        adaptive_confidence_scale=4.0,
        update_count=1,
        episodes_since_update=0,
        entries=(
            TrackSamplingRuntimeEntry(
                track_id="short",
                course_key="short",
                label="Short",
                base_weight=1.0,
                current_weight=1.0,
                completed_frames=100,
                episode_count=1,
                finished_episode_count=0,
                success_sample_count=1,
                ema_episode_frames=100.0,
                ema_completion_fraction=0.2,
            ),
            TrackSamplingRuntimeEntry(
                track_id="long",
                course_key="long",
                label="Long",
                base_weight=1.0,
                current_weight=1.0,
                completed_frames=400,
                episode_count=1,
                finished_episode_count=0,
                success_sample_count=1,
                ema_episode_frames=400.0,
                ema_completion_fraction=0.2,
            ),
        ),
    )

    controller = StepBalancedTrackSamplingController(
        resolved_courses=resolved_track_sampling_courses({"short": 1.0, "long": 1.0}),
        action_repeat=2,
        update_episodes=2,
        ema_alpha=1.0,
        max_weight_scale=5.0,
        restored_state=restored,
    )

    weights = controller.current_weights()

    assert weights == pytest.approx({"short": 1.92, "long": 0.08})


def test_step_balance_controller_keeps_over_budget_courses_sampleable() -> None:
    restored = TrackSamplingRuntimeState(
        sampling_mode="adaptive_step_balanced",
        action_repeat=1,
        update_episodes=2,
        ema_alpha=1.0,
        max_weight_scale=5.0,
        adaptive_completion_weight=0.35,
        adaptive_target_completion=0.9,
        adaptive_min_confidence_episodes=24,
        adaptive_confidence_scale=4.0,
        update_count=1,
        episodes_since_update=0,
        entries=(
            TrackSamplingRuntimeEntry(
                track_id="failed_over_budget",
                course_key="failed_over_budget",
                label="Failed Over Budget",
                base_weight=1.0,
                current_weight=1.0,
                completed_frames=400,
                episode_count=1,
                finished_episode_count=0,
                success_sample_count=1,
                ema_episode_frames=400.0,
                ema_completion_fraction=0.0,
            ),
            TrackSamplingRuntimeEntry(
                track_id="under_budget",
                course_key="under_budget",
                label="Under Budget",
                base_weight=1.0,
                current_weight=1.0,
                completed_frames=100,
                episode_count=1,
                finished_episode_count=0,
                success_sample_count=1,
                ema_episode_frames=100.0,
                ema_completion_fraction=0.0,
            ),
        ),
    )

    controller = StepBalancedTrackSamplingController(
        resolved_courses=resolved_track_sampling_courses(
            {"failed_over_budget": 1.0, "under_budget": 1.0}
        ),
        sampling_mode="adaptive_step_balanced",
        action_repeat=1,
        update_episodes=2,
        ema_alpha=1.0,
        max_weight_scale=5.0,
        adaptive_completion_weight=0.35,
        adaptive_target_completion=0.9,
        restored_state=restored,
    )

    weights = controller.current_weights()

    assert weights["under_budget"] > weights["failed_over_budget"]
    assert weights["failed_over_budget"] > 0.0


def test_step_balance_controller_uses_steady_state_weights_when_no_course_has_debt() -> None:
    restored = TrackSamplingRuntimeState(
        sampling_mode="step_balanced",
        action_repeat=1,
        update_episodes=2,
        ema_alpha=1.0,
        max_weight_scale=5.0,
        adaptive_completion_weight=0.35,
        adaptive_target_completion=0.9,
        adaptive_min_confidence_episodes=24,
        adaptive_confidence_scale=4.0,
        update_count=1,
        episodes_since_update=0,
        entries=(
            TrackSamplingRuntimeEntry(
                track_id="short",
                course_key="short",
                label="Short",
                base_weight=1.0,
                current_weight=1.0,
                completed_frames=100,
                episode_count=1,
                finished_episode_count=0,
                success_sample_count=1,
                ema_episode_frames=100.0,
                ema_completion_fraction=0.2,
            ),
            TrackSamplingRuntimeEntry(
                track_id="long",
                course_key="long",
                label="Long",
                base_weight=1.0,
                current_weight=1.0,
                completed_frames=100,
                episode_count=1,
                finished_episode_count=0,
                success_sample_count=1,
                ema_episode_frames=400.0,
                ema_completion_fraction=0.2,
            ),
        ),
    )

    controller = StepBalancedTrackSamplingController(
        resolved_courses=resolved_track_sampling_courses({"short": 1.0, "long": 1.0}),
        action_repeat=1,
        update_episodes=2,
        ema_alpha=1.0,
        max_weight_scale=5.0,
        restored_state=restored,
    )

    weights = controller.current_weights()

    assert weights["short"] > weights["long"]
    assert weights["short"] * 100 == pytest.approx(weights["long"] * 400)
