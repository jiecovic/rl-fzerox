# tests/core/training/test_track_sampling_balance.py
from pathlib import Path

import pytest

from rl_fzerox.apps.run_manager.api.payloads import track_sampling_state_payload
from rl_fzerox.core.domain.x_cup import X_CUP_COURSE, generated_x_cup_slot_key
from rl_fzerox.core.runtime_spec.schema import (
    CurriculumConfig,
    EnvConfig,
    TrackSamplingConfig,
    TrackSamplingEntryConfig,
)
from rl_fzerox.core.training.session.callbacks.track_sampling import (
    StepBalancedTrackSamplingController,
    TrackSamplingRuntimeEntry,
    TrackSamplingRuntimeState,
    load_track_sampling_runtime_state,
    replace_runtime_generation,
    save_track_sampling_runtime_state,
)
from rl_fzerox.core.training.session.callbacks.track_sampling.deficit import (
    DEFICIT_QUEUE_SETTINGS,
    DeficitBudgetSettings,
    DeficitBudgetTrackSamplingController,
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


def test_fixed_env_controller_tracks_runtime_stats_without_reweighting(tmp_path: Path) -> None:
    controller = StepBalancedTrackSamplingController(
        track_base_weights={"mute": 1.0, "silence": 1.0},
        sampling_mode="fixed_env",
        action_repeat=2,
        update_episodes=2,
        ema_alpha=1.0,
        max_weight_scale=5.0,
        log_details=True,
    )

    weights = controller.record_episodes(
        (
            {
                "track_id": "mute",
                "episode_step": 100,
                "episode_completion_fraction": 1.0,
                "termination_reason": "finished",
            },
            {
                "track_id": "silence",
                "episode_step": 400,
                "episode_completion_fraction": 0.25,
                "termination_reason": "stalled",
            },
        )
    )

    assert weights == pytest.approx({"mute": 1.0, "silence": 1.0})
    assert controller.log_values() == {}
    runtime = controller.runtime_state()
    assert runtime.sampling_mode == "fixed_env"
    assert {entry.course_key: entry.completed_frames for entry in runtime.entries} == {
        "mute": 100,
        "silence": 400,
    }
    payload = track_sampling_state_payload(runtime)
    raw_payload_entries = payload["entries"]
    assert isinstance(raw_payload_entries, list)
    payload_entries: dict[str, dict[str, object]] = {}
    for entry in raw_payload_entries:
        assert isinstance(entry, dict)
        course_key = entry.get("course_key")
        assert isinstance(course_key, str)
        payload_entries[course_key] = entry
    assert payload_entries["mute"]["success_rate"] == pytest.approx(1.0)
    assert payload_entries["silence"]["ema_completion_fraction"] == pytest.approx(0.25)

    state_path = tmp_path / "track_sampling_state.json"
    save_track_sampling_runtime_state(state_path, runtime)
    restored = load_track_sampling_runtime_state(state_path)

    assert restored is not None
    assert restored.sampling_mode == "fixed_env"


def test_deficit_budget_payload_uses_uniform_adaptive_target_mix() -> None:
    state = TrackSamplingRuntimeState(
        sampling_mode="deficit_budget",
        action_repeat=2,
        update_episodes=20,
        ema_alpha=0.02,
        max_weight_scale=3.0,
        adaptive_completion_weight=0.3,
        adaptive_target_completion=1.0,
        adaptive_min_confidence_episodes=1,
        adaptive_confidence_scale=3.0,
        update_count=1,
        episodes_since_update=0,
        entries=(
            TrackSamplingRuntimeEntry(
                track_id="easy",
                course_key="easy",
                label="Easy",
                base_weight=1.0,
                current_weight=1.0,
                completed_frames=0,
                episode_count=0,
                finished_episode_count=0,
                success_sample_count=0,
                ema_episode_frames=None,
                ema_completion_fraction=None,
            ),
            TrackSamplingRuntimeEntry(
                track_id="hard",
                course_key="hard",
                label="Hard",
                base_weight=1.0,
                current_weight=3.0,
                completed_frames=0,
                episode_count=0,
                finished_episode_count=0,
                success_sample_count=0,
                ema_episode_frames=None,
                ema_completion_fraction=None,
            ),
        ),
    )

    payload = track_sampling_state_payload(state)
    entries = {
        entry["course_key"]: entry
        for entry in payload["entries"]
        if isinstance(entry, dict) and isinstance(entry.get("course_key"), str)
    }

    assert entries["easy"]["target_step_share"] == pytest.approx(0.425)
    assert entries["hard"]["target_step_share"] == pytest.approx(0.575)


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


def test_step_balance_runtime_state_exposes_generated_x_cup_generation() -> None:
    slot_key = generated_x_cup_slot_key(2)
    entry = TrackSamplingEntryConfig(
        id="x_cup_slot_blue_falcon",
        course_id="x_cup_abcd1234",
        runtime_course_key=slot_key,
        course_name="X Cup abcd1234",
        mode=X_CUP_COURSE.race_mode,
        course_index=X_CUP_COURSE.course_index,
        generated_course_kind=X_CUP_COURSE.generated_kind,
        generated_course_seed=123,
        generated_course_hash="abcd1234",
        generated_course_slot=2,
        generated_course_generation=3,
    )
    controller = StepBalancedTrackSamplingController.from_configs(
        env_config=EnvConfig(
            action_repeat=2,
            track_sampling=TrackSamplingConfig(
                enabled=True,
                sampling_mode="step_balanced",
                entries=(
                    entry,
                    TrackSamplingEntryConfig(
                        id="mute_city_blue_falcon",
                        course_id="mute_city",
                        course_name="Mute City",
                    ),
                ),
            ),
        ),
        curriculum_config=CurriculumConfig(),
    )

    assert controller is not None
    x_cup_runtime = next(
        entry for entry in controller.runtime_state().entries if entry.course_key == slot_key
    )

    assert x_cup_runtime.track_id == slot_key
    assert x_cup_runtime.generated_course_slot == 2
    assert x_cup_runtime.generated_course_generation == 3


def test_track_sampling_payload_uses_runtime_generated_x_cup_metadata() -> None:
    slot_key = generated_x_cup_slot_key(1)
    state = TrackSamplingRuntimeState(
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
                track_id=slot_key,
                course_key=slot_key,
                label="X Cup abcd1234",
                base_weight=1.0,
                current_weight=1.0,
                completed_frames=100,
                episode_count=1,
                finished_episode_count=0,
                success_sample_count=1,
                ema_episode_frames=100.0,
                ema_completion_fraction=0.2,
                generated_course_slot=1,
                generated_course_generation=4,
            ),
        ),
    )

    payload = track_sampling_state_payload(state)

    entries_payload = payload["entries"]
    assert isinstance(entries_payload, list)
    entry_payload = entries_payload[0]
    assert isinstance(entry_payload, dict)
    assert entry_payload["generated_course_slot"] == 1
    assert entry_payload["generated_course_generation"] == 4


def test_replace_runtime_generation_keeps_slot_history_and_resets_current_generation() -> None:
    slot_key = generated_x_cup_slot_key(0)
    state = TrackSamplingRuntimeState(
        sampling_mode="adaptive_step_balanced",
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
                track_id=slot_key,
                course_key=slot_key,
                label="X Cup old",
                base_weight=1.0,
                current_weight=1.0,
                completed_frames=5000,
                episode_count=20,
                finished_episode_count=18,
                success_sample_count=20,
                ema_episode_frames=250.0,
                ema_completion_fraction=0.95,
                generation_episode_count=20,
                generation_finished_episode_count=18,
                generation_success_sample_count=20,
                generation_ema_completion_fraction=0.95,
                generated_course_slot=0,
                generated_course_generation=1,
            ),
        ),
    )

    replaced = replace_runtime_generation(
        state,
        course_key=slot_key,
        replacement_label="X Cup new",
        generated_course_slot=0,
        generated_course_generation=2,
        generated_entry_id="x_cup_new_gp_race",
        generated_course_id="x_cup_new",
        generated_course_name="X Cup new",
        generated_course_hash="new",
        generated_course_seed=123,
        generated_baseline_state_path="/tmp/x_cup_new.state",
        generated_course_segment_count=256,
        generated_course_length=12345.0,
    )
    entry = replaced.entries[0]

    assert entry.course_key == slot_key
    assert entry.completed_frames == 5000
    assert entry.ema_episode_frames == 250.0
    assert entry.episode_count == 20
    assert entry.success_sample_count == 20
    assert entry.ema_completion_fraction == 0.95
    assert entry.generation_episode_count == 0
    assert entry.generation_success_sample_count == 0
    assert entry.generation_ema_completion_fraction is None
    assert entry.generated_course_generation == 2


def test_adaptive_step_balance_controller_tilts_weight_toward_lower_completion() -> None:
    controller = StepBalancedTrackSamplingController(
        track_base_weights={"mute": 1.0, "silence": 1.0},
        sampling_mode="adaptive_step_balanced",
        action_repeat=2,
        update_episodes=2,
        ema_alpha=1.0,
        max_weight_scale=5.0,
        adaptive_completion_weight=0.5,
        adaptive_target_completion=0.9,
    )

    weights = controller.record_episodes(
        (
            {"track_id": "mute", "episode_step": 200, "episode_completion_fraction": 0.2},
            {"track_id": "silence", "episode_step": 200, "episode_completion_fraction": 0.8},
        )
    )

    assert weights is not None
    assert weights["mute"] > weights["silence"]


def test_adaptive_step_balance_controller_uses_finish_rate_when_completion_is_similar() -> None:
    controller = StepBalancedTrackSamplingController(
        track_base_weights={"sector_alpha": 1.0, "white_land": 1.0},
        sampling_mode="adaptive_step_balanced",
        action_repeat=2,
        update_episodes=2,
        ema_alpha=1.0,
        max_weight_scale=5.0,
        adaptive_completion_weight=0.35,
        adaptive_target_completion=0.9,
    )

    weights = controller.record_episodes(
        (
            {
                "track_id": "sector_alpha",
                "episode_step": 200,
                "episode_completion_fraction": 0.95,
                "termination_reason": "finished",
            },
            {
                "track_id": "white_land",
                "episode_step": 200,
                "episode_completion_fraction": 0.85,
                "termination_reason": "stalled",
            },
        )
    )

    assert weights is not None
    assert weights["white_land"] > weights["sector_alpha"]


def test_adaptive_step_balance_controller_can_create_large_target_spread() -> None:
    controller = StepBalancedTrackSamplingController(
        track_base_weights={"easy": 1.0, "hard": 1.0},
        sampling_mode="adaptive_step_balanced",
        action_repeat=2,
        update_episodes=2,
        ema_alpha=1.0,
        max_weight_scale=5.0,
        adaptive_completion_weight=0.35,
        adaptive_target_completion=0.9,
    )

    controller.record_episodes(
        (
            {
                "track_id": "easy",
                "episode_step": 200,
                "episode_completion_fraction": 1.0,
                "termination_reason": "finished",
            },
            {
                "track_id": "hard",
                "episode_step": 200,
                "episode_completion_fraction": 0.0,
                "termination_reason": "stalled",
            },
        )
    )

    runtime = controller.runtime_state()
    weights_by_track = {entry.track_id: entry.current_weight for entry in runtime.entries}

    assert weights_by_track["hard"] > 1.5 * weights_by_track["easy"]


def test_adaptive_step_balance_prioritizes_low_confidence_courses() -> None:
    controller = StepBalancedTrackSamplingController(
        track_base_weights={"known": 1.0, "unknown": 1.0},
        sampling_mode="adaptive_step_balanced",
        action_repeat=1,
        update_episodes=2,
        ema_alpha=1.0,
        max_weight_scale=5.0,
        adaptive_completion_weight=0.0,
        adaptive_target_completion=0.9,
        adaptive_min_confidence_episodes=4,
        adaptive_confidence_scale=5.0,
    )

    weights = controller.record_episodes(
        (
            {
                "track_id": "known",
                "episode_step": 100,
                "episode_completion_fraction": 1.0,
                "termination_reason": "finished",
            },
            {
                "track_id": "unknown",
                "episode_step": 100,
                "episode_completion_fraction": 1.0,
                "termination_reason": "finished",
            },
            {
                "track_id": "known",
                "episode_step": 100,
                "episode_completion_fraction": 1.0,
                "termination_reason": "finished",
            },
            {
                "track_id": "known",
                "episode_step": 100,
                "episode_completion_fraction": 1.0,
                "termination_reason": "finished",
            },
            {
                "track_id": "known",
                "episode_step": 100,
                "episode_completion_fraction": 1.0,
                "termination_reason": "finished",
            },
        )
    )

    assert weights is not None
    assert weights["unknown"] > weights["known"]


def test_adaptive_step_balance_converts_frame_target_to_reset_weight() -> None:
    controller = StepBalancedTrackSamplingController(
        track_base_weights={"easy_short": 1.0, "hard_long": 1.0},
        sampling_mode="adaptive_step_balanced",
        action_repeat=2,
        update_episodes=2,
        ema_alpha=1.0,
        max_weight_scale=20.0,
        adaptive_completion_weight=1.0,
        adaptive_target_completion=0.9,
    )

    controller.record_episodes(
        (
            {
                "track_id": "easy_short",
                "episode_step": 100,
                "episode_completion_fraction": 1.0,
                "termination_reason": "finished",
            },
            {
                "track_id": "hard_long",
                "episode_step": 400,
                "episode_completion_fraction": 0.0,
                "termination_reason": "stalled",
            },
        )
    )

    runtime = controller.runtime_state()
    entries = {entry.track_id: entry for entry in runtime.entries}
    easy = entries["easy_short"]
    hard = entries["hard_long"]

    assert hard.current_weight > easy.current_weight
    assert hard.current_weight * 400 > easy.current_weight * 100 * 10.0


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


def test_step_balance_controller_can_suppress_generated_course_logs() -> None:
    controller = StepBalancedTrackSamplingController.from_configs(
        env_config=EnvConfig(
            track_sampling=TrackSamplingConfig(
                enabled=True,
                sampling_mode="step_balanced",
                entries=(
                    TrackSamplingEntryConfig(
                        id="x_cup_a_blue_falcon_balanced",
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
                        id="x_cup_b_blue_falcon_balanced",
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
        curriculum_config=CurriculumConfig(),
    )

    assert controller is not None
    weights = controller.record_episodes(
        (
            {"track_id": "x_cup_a_blue_falcon_balanced", "episode_step": 100},
            {"track_id": "x_cup_b_blue_falcon_balanced", "episode_step": 300},
        )
    )

    assert weights is not None
    assert set(weights) == {"x_cup_a_blue_falcon_balanced", "x_cup_b_blue_falcon_balanced"}
    assert controller.log_values() == {}


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
    assert restored_state.sampling_mode == "step_balanced"
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
        track_base_weights={"short": 1.0, "long": 1.0},
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
        track_base_weights={"failed_over_budget": 1.0, "under_budget": 1.0},
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
        track_base_weights={"short": 1.0, "long": 1.0},
        action_repeat=1,
        update_episodes=2,
        ema_alpha=1.0,
        max_weight_scale=5.0,
        restored_state=restored,
    )

    weights = controller.current_weights()

    assert weights["short"] > weights["long"]
    assert weights["short"] * 100 == pytest.approx(weights["long"] * 400)


def test_deficit_budget_controller_reserves_queue_assignments_fairly() -> None:
    controller = DeficitBudgetTrackSamplingController(
        track_base_weights={"mute": 1.0, "silence": 1.0},
        action_repeat=1,
        settings=DeficitBudgetSettings(
            uniform_fraction=1.0,
            min_weight=1.0,
            max_weight=3.0,
            ema_alpha=0.02,
            weight_update_rollouts=20,
        ),
        track_course_keys={"mute": "mute", "silence": "silence"},
        track_log_keys={"mute": "mute", "silence": "silence"},
        track_labels={"mute": "Mute City", "silence": "Silence"},
        track_log_enabled={"mute": True, "silence": True},
        seed=7,
    )

    controller.add_rollout_budget(total_steps=200)
    assignments = tuple(controller.next_course_key(assignment_cost=100.0) for _ in range(4))

    assert assignments.count("mute") == 2
    assert assignments.count("silence") == 2


def test_deficit_budget_controller_refills_bounded_balanced_queues() -> None:
    controller = DeficitBudgetTrackSamplingController(
        track_base_weights={"mute": 1.0, "silence": 1.0},
        action_repeat=1,
        settings=DeficitBudgetSettings(
            uniform_fraction=1.0,
            min_weight=1.0,
            max_weight=3.0,
            ema_alpha=0.02,
            weight_update_rollouts=20,
        ),
        track_course_keys={"mute": "mute", "silence": "silence"},
        track_log_keys={"mute": "mute", "silence": "silence"},
        track_labels={"mute": "Mute City", "silence": "Silence"},
        track_log_enabled={"mute": True, "silence": True},
        seed=7,
    )

    controller.add_rollout_budget(total_steps=200)
    refills = controller.refill_queues((0, 99), rollout_steps=100)

    assert set(refills) == {0}
    assert len(refills[0]) == DEFICIT_QUEUE_SETTINGS.minimum_refill_size
    assert abs(refills[0].count("mute") - refills[0].count("silence")) <= 1


def test_deficit_budget_controller_prefers_courses_with_positive_step_debt() -> None:
    controller = DeficitBudgetTrackSamplingController(
        track_base_weights={"mute": 1.0, "silence": 1.0},
        action_repeat=1,
        settings=DeficitBudgetSettings(
            uniform_fraction=1.0,
            min_weight=1.0,
            max_weight=3.0,
            ema_alpha=0.02,
            weight_update_rollouts=20,
        ),
        track_course_keys={"mute": "mute", "silence": "silence"},
        track_log_keys={"mute": "mute", "silence": "silence"},
        track_labels={"mute": "Mute City", "silence": "Silence"},
        track_log_enabled={"mute": True, "silence": True},
        seed=7,
    )

    controller.add_rollout_budget(total_steps=200)
    controller.record_step_infos(({"track_id": "mute"},) * 75)

    assert controller.next_course_key(assignment_cost=1.0) == "silence"


def test_deficit_budget_runtime_state_persists_accounted_step_totals() -> None:
    controller = DeficitBudgetTrackSamplingController(
        track_base_weights={"mute": 1.0, "silence": 1.0},
        action_repeat=2,
        settings=DeficitBudgetSettings(
            uniform_fraction=1.0,
            min_weight=1.0,
            max_weight=3.0,
            ema_alpha=0.02,
            weight_update_rollouts=20,
        ),
        track_course_keys={"mute": "mute", "silence": "silence"},
        track_log_keys={"mute": "mute", "silence": "silence"},
        track_labels={"mute": "Mute City", "silence": "Silence"},
        track_log_enabled={"mute": True, "silence": True},
        seed=7,
    )

    controller.record_step_infos(({"track_id": "mute"}, {"track_id": "mute"}))
    controller.record_step_infos(({"track_id": "silence"},))

    runtime = controller.runtime_state()

    assert {entry.course_key: entry.completed_frames for entry in runtime.entries} == {
        "mute": 4,
        "silence": 2,
    }
    assert {entry.course_key: entry.episode_count for entry in runtime.entries} == {
        "mute": 0,
        "silence": 0,
    }


def test_deficit_budget_controller_raises_target_share_for_problem_course() -> None:
    controller = DeficitBudgetTrackSamplingController(
        track_base_weights={"easy": 1.0, "hard": 1.0},
        action_repeat=1,
        settings=DeficitBudgetSettings(
            uniform_fraction=0.7,
            min_weight=1.0,
            max_weight=3.0,
            ema_alpha=1.0,
            weight_update_rollouts=1,
        ),
        track_course_keys={"easy": "easy", "hard": "hard"},
        track_log_keys={"easy": "easy", "hard": "hard"},
        track_labels={"easy": "Easy", "hard": "Hard"},
        track_log_enabled={"easy": True, "hard": True},
        seed=7,
    )

    controller.record_episodes(
        (
            {
                "track_id": "easy",
                "episode_step": 100,
                "episode_completion_fraction": 1.0,
                "termination_reason": "finished",
            },
            {
                "track_id": "hard",
                "episode_step": 100,
                "episode_completion_fraction": 0.25,
                "termination_reason": "crashed",
            },
        )
    )
    controller.maybe_update_weights()
    values = controller.log_values()

    assert values["track_sampling/hard/problem_ema"] > values["track_sampling/easy/problem_ema"]
    assert (
        values["track_sampling/hard/target_step_share"]
        > values["track_sampling/easy/target_step_share"]
    )
    assert values["track_sampling/hard/target_step_share"] < 0.7


def test_deficit_budget_controller_restores_runtime_stats() -> None:
    restored = TrackSamplingRuntimeState(
        sampling_mode="deficit_budget",
        action_repeat=1,
        update_episodes=20,
        ema_alpha=0.02,
        max_weight_scale=3.0,
        adaptive_completion_weight=0.3,
        adaptive_target_completion=1.0,
        adaptive_min_confidence_episodes=1,
        adaptive_confidence_scale=3.0,
        update_count=2,
        episodes_since_update=3,
        entries=(
            TrackSamplingRuntimeEntry(
                track_id="mute",
                course_key="mute",
                label="Mute City",
                base_weight=1.0,
                current_weight=1.0,
                completed_frames=300,
                episode_count=3,
                finished_episode_count=1,
                success_sample_count=3,
                ema_episode_frames=100.0,
                ema_completion_fraction=0.5,
            ),
            TrackSamplingRuntimeEntry(
                track_id="silence",
                course_key="silence",
                label="Silence",
                base_weight=1.0,
                current_weight=1.0,
                completed_frames=100,
                episode_count=1,
                finished_episode_count=0,
                success_sample_count=1,
                ema_episode_frames=100.0,
                ema_completion_fraction=0.25,
            ),
        ),
    )
    controller = DeficitBudgetTrackSamplingController(
        track_base_weights={"mute": 1.0, "silence": 1.0},
        action_repeat=1,
        settings=DeficitBudgetSettings(
            uniform_fraction=0.7,
            min_weight=1.0,
            max_weight=3.0,
            ema_alpha=0.02,
            weight_update_rollouts=20,
        ),
        track_course_keys={"mute": "mute", "silence": "silence"},
        track_log_keys={"mute": "mute", "silence": "silence"},
        track_labels={"mute": "Mute City", "silence": "Silence"},
        track_log_enabled={"mute": True, "silence": True},
        restored_state=restored,
        seed=7,
    )

    runtime = controller.runtime_state()

    assert runtime.sampling_mode == "deficit_budget"
    assert runtime.update_count == 2
    assert {entry.course_key: entry.completed_frames for entry in runtime.entries} == {
        "mute": 300,
        "silence": 100,
    }
    values = controller.log_values()
    assert (
        values["track_sampling/silence/target_step_share"]
        > values["track_sampling/mute/target_step_share"]
    )
