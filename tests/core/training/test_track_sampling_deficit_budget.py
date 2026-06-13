# tests/core/training/test_track_sampling_deficit_budget.py
from pathlib import Path

import pytest

from rl_fzerox.apps.run_manager.api.payloads.track_sampling import (
    track_sampling_state_payload,
)
from rl_fzerox.core.domain.x_cup import generated_x_cup_slot_key
from rl_fzerox.core.training.session.callbacks.track_sampling import (
    StepBalancedTrackSamplingController,
    TrackSamplingRuntimeEntry,
    TrackSamplingRuntimeState,
    load_track_sampling_runtime_state,
    save_track_sampling_runtime_state,
)
from rl_fzerox.core.training.session.callbacks.track_sampling.deficit import (
    DEFICIT_QUEUE_SETTINGS,
    DeficitBudgetSettings,
    DeficitBudgetTrackSamplingController,
)
from tests.core.training.track_sampling_support import resolved_track_sampling_courses


def test_fixed_env_controller_tracks_runtime_stats_without_reweighting(tmp_path: Path) -> None:
    controller = StepBalancedTrackSamplingController(
        resolved_courses=resolved_track_sampling_courses({"mute": 1.0, "silence": 1.0}),
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
    payload_entries = payload["entries"]
    assert isinstance(payload_entries, list)
    entries = {
        entry["course_key"]: entry
        for entry in payload_entries
        if isinstance(entry, dict) and isinstance(entry.get("course_key"), str)
    }

    assert entries["easy"]["target_step_share"] == pytest.approx(0.425)
    assert entries["hard"]["target_step_share"] == pytest.approx(0.575)


def test_deficit_budget_controller_reserves_queue_assignments_fairly() -> None:
    controller = DeficitBudgetTrackSamplingController(
        resolved_courses=resolved_track_sampling_courses(
            {"mute": 1.0, "silence": 1.0},
            course_keys={"mute": "mute", "silence": "silence"},
            log_keys={"mute": "mute", "silence": "silence"},
            labels={"mute": "Mute City", "silence": "Silence"},
            log_enabled={"mute": True, "silence": True},
        ),
        action_repeat=1,
        settings=DeficitBudgetSettings(
            uniform_fraction=1.0,
            min_weight=1.0,
            max_weight=3.0,
            ema_alpha=0.02,
            weight_update_rollouts=20,
        ),
        seed=7,
    )

    controller.add_rollout_budget(total_steps=200)
    assignments = tuple(
        controller.next_course_key(fallback_assignment_steps=100.0) for _ in range(4)
    )

    assert assignments.count("mute") == 2
    assert assignments.count("silence") == 2


def test_deficit_budget_controller_refills_bounded_balanced_queues() -> None:
    controller = DeficitBudgetTrackSamplingController(
        resolved_courses=resolved_track_sampling_courses(
            {"mute": 1.0, "silence": 1.0},
            course_keys={"mute": "mute", "silence": "silence"},
            log_keys={"mute": "mute", "silence": "silence"},
            labels={"mute": "Mute City", "silence": "Silence"},
            log_enabled={"mute": True, "silence": True},
        ),
        action_repeat=1,
        settings=DeficitBudgetSettings(
            uniform_fraction=1.0,
            min_weight=1.0,
            max_weight=3.0,
            ema_alpha=0.02,
            weight_update_rollouts=20,
        ),
        seed=7,
    )

    controller.add_rollout_budget(total_steps=200)
    refills = controller.refill_queues((0, 99), fallback_assignment_steps=100)

    assert set(refills) == {0}
    assert len(refills[0]) == DEFICIT_QUEUE_SETTINGS.minimum_refill_size
    assert abs(refills[0].count("mute") - refills[0].count("silence")) <= 1


def test_deficit_budget_controller_prefers_courses_with_positive_step_debt() -> None:
    controller = DeficitBudgetTrackSamplingController(
        resolved_courses=resolved_track_sampling_courses(
            {"mute": 1.0, "silence": 1.0},
            course_keys={"mute": "mute", "silence": "silence"},
            log_keys={"mute": "mute", "silence": "silence"},
            labels={"mute": "Mute City", "silence": "Silence"},
            log_enabled={"mute": True, "silence": True},
        ),
        action_repeat=1,
        settings=DeficitBudgetSettings(
            uniform_fraction=1.0,
            min_weight=1.0,
            max_weight=3.0,
            ema_alpha=0.02,
            weight_update_rollouts=20,
        ),
        seed=7,
    )

    controller.add_rollout_budget(total_steps=200)
    controller.record_step_infos(({"track_id": "mute"},) * 75)

    assert controller.next_course_key(fallback_assignment_steps=1.0) == "silence"


def test_deficit_budget_controller_keeps_alt_baselines_out_of_runtime_stats() -> None:
    controller = DeficitBudgetTrackSamplingController(
        resolved_courses=resolved_track_sampling_courses(
            {"mute": 1.0, "silence": 1.0},
            course_keys={"mute": "mute", "silence": "silence"},
            log_keys={"mute": "mute", "silence": "silence"},
            labels={"mute": "Mute City", "silence": "Silence"},
            log_enabled={"mute": True, "silence": True},
        ),
        action_repeat=2,
        settings=DeficitBudgetSettings(
            uniform_fraction=1.0,
            min_weight=1.0,
            max_weight=3.0,
            ema_alpha=1.0,
            weight_update_rollouts=20,
        ),
        seed=7,
    )

    controller.add_rollout_budget(total_steps=200)
    controller.record_step_infos(({"track_id": "mute", "track_alt_baseline_id": "alt-a"},) * 40)
    controller.record_episodes(
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
    assert {entry.course_key: entry.completed_frames for entry in runtime.entries} == {
        "mute": 0,
        "silence": 0,
    }
    assert {entry.course_key: entry.episode_count for entry in runtime.entries} == {
        "mute": 0,
        "silence": 0,
    }
    assert controller.next_course_key(fallback_assignment_steps=1.0) == "silence"


def test_deficit_budget_controller_restores_historical_step_debt() -> None:
    restored = TrackSamplingRuntimeState(
        sampling_mode="deficit_budget",
        action_repeat=1,
        update_episodes=20,
        ema_alpha=0.02,
        max_weight_scale=3.0,
        adaptive_completion_weight=0.0,
        adaptive_target_completion=1.0,
        adaptive_min_confidence_episodes=1,
        adaptive_confidence_scale=3.0,
        update_count=1,
        episodes_since_update=0,
        entries=(
            TrackSamplingRuntimeEntry(
                track_id="balanced",
                course_key="balanced",
                label="Balanced",
                base_weight=1.0,
                current_weight=1.0,
                completed_frames=1000,
                episode_count=10,
                finished_episode_count=10,
                success_sample_count=10,
                ema_episode_frames=100.0,
                ema_completion_fraction=1.0,
            ),
            TrackSamplingRuntimeEntry(
                track_id="crash",
                course_key="crash",
                label="Crash",
                base_weight=1.0,
                current_weight=1.0,
                completed_frames=100,
                episode_count=10,
                finished_episode_count=0,
                success_sample_count=10,
                ema_episode_frames=10.0,
                ema_completion_fraction=0.1,
            ),
        ),
    )
    controller = DeficitBudgetTrackSamplingController(
        resolved_courses=resolved_track_sampling_courses(
            {"balanced": 1.0, "crash": 1.0},
            course_keys={"balanced": "balanced", "crash": "crash"},
            log_keys={"balanced": "balanced", "crash": "crash"},
            labels={"balanced": "Balanced", "crash": "Crash"},
            log_enabled={"balanced": True, "crash": True},
        ),
        action_repeat=1,
        settings=DeficitBudgetSettings(
            uniform_fraction=1.0,
            min_weight=1.0,
            max_weight=3.0,
            ema_alpha=0.02,
            weight_update_rollouts=20,
        ),
        restored_state=restored,
        seed=7,
    )

    assignments = tuple(
        controller.next_course_key(fallback_assignment_steps=100.0) for _ in range(8)
    )

    assert assignments == ("crash",) * 8


def test_deficit_budget_controller_uses_episode_ema_as_assignment_cost() -> None:
    controller = DeficitBudgetTrackSamplingController(
        resolved_courses=resolved_track_sampling_courses(
            {"long": 1.0, "short": 1.0},
            course_keys={"long": "long", "short": "short"},
            log_keys={"long": "long", "short": "short"},
            labels={"long": "Long", "short": "Short"},
            log_enabled={"long": True, "short": True},
        ),
        action_repeat=1,
        settings=DeficitBudgetSettings(
            uniform_fraction=1.0,
            min_weight=1.0,
            max_weight=3.0,
            ema_alpha=1.0,
            weight_update_rollouts=20,
        ),
        seed=7,
    )

    controller.record_episodes(
        (
            {
                "track_id": "long",
                "episode_step": 100,
                "episode_completion_fraction": 1.0,
                "termination_reason": "finished",
            },
            {
                "track_id": "short",
                "episode_step": 10,
                "episode_completion_fraction": 0.1,
                "termination_reason": "crashed",
            },
        )
    )
    controller.add_rollout_budget(total_steps=200)
    controller.record_step_infos(({"track_id": "long"},) * 100)

    assignments = tuple(
        controller.next_course_key(fallback_assignment_steps=100.0) for _ in range(5)
    )

    assert assignments == ("short",) * 5


def test_deficit_budget_runtime_state_persists_accounted_step_totals() -> None:
    controller = DeficitBudgetTrackSamplingController(
        resolved_courses=resolved_track_sampling_courses(
            {"mute": 1.0, "silence": 1.0},
            course_keys={"mute": "mute", "silence": "silence"},
            log_keys={"mute": "mute", "silence": "silence"},
            labels={"mute": "Mute City", "silence": "Silence"},
            log_enabled={"mute": True, "silence": True},
        ),
        action_repeat=2,
        settings=DeficitBudgetSettings(
            uniform_fraction=1.0,
            min_weight=1.0,
            max_weight=3.0,
            ema_alpha=0.02,
            weight_update_rollouts=20,
        ),
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
        resolved_courses=resolved_track_sampling_courses(
            {"easy": 1.0, "hard": 1.0},
            course_keys={"easy": "easy", "hard": "hard"},
            log_keys={"easy": "easy", "hard": "hard"},
            labels={"easy": "Easy", "hard": "Hard"},
            log_enabled={"easy": True, "hard": True},
        ),
        action_repeat=1,
        settings=DeficitBudgetSettings(
            uniform_fraction=0.7,
            min_weight=1.0,
            max_weight=3.0,
            ema_alpha=1.0,
            weight_update_rollouts=1,
        ),
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
        resolved_courses=resolved_track_sampling_courses(
            {"mute": 1.0, "silence": 1.0},
            course_keys={"mute": "mute", "silence": "silence"},
            log_keys={"mute": "mute", "silence": "silence"},
            labels={"mute": "Mute City", "silence": "Silence"},
            log_enabled={"mute": True, "silence": True},
        ),
        action_repeat=1,
        settings=DeficitBudgetSettings(
            uniform_fraction=0.7,
            min_weight=1.0,
            max_weight=3.0,
            ema_alpha=0.02,
            weight_update_rollouts=20,
        ),
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


def test_deficit_budget_controller_backfills_first_x_cup_generation_stats() -> None:
    slot_key = generated_x_cup_slot_key(0)
    restored = TrackSamplingRuntimeState(
        sampling_mode="deficit_budget",
        action_repeat=2,
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
                track_id=slot_key,
                course_key=slot_key,
                label="X Cup abcd1234",
                base_weight=1.0,
                current_weight=1.0,
                completed_frames=171660,
                episode_count=40,
                finished_episode_count=10,
                success_sample_count=40,
                ema_episode_frames=4000.0,
                ema_completion_fraction=0.686,
                generated_course_slot=0,
                generated_course_generation=1,
            ),
            TrackSamplingRuntimeEntry(
                track_id="mute",
                course_key="mute",
                label="Mute City",
                base_weight=1.0,
                current_weight=1.0,
                completed_frames=200,
                episode_count=1,
                finished_episode_count=0,
                success_sample_count=1,
                ema_episode_frames=200.0,
                ema_completion_fraction=0.25,
            ),
        ),
    )
    controller = DeficitBudgetTrackSamplingController(
        resolved_courses=resolved_track_sampling_courses(
            {"x_cup_entry": 1.0, "mute": 1.0},
            course_keys={"x_cup_entry": slot_key, "mute": "mute"},
            log_keys={"x_cup_entry": slot_key, "mute": "mute"},
            labels={slot_key: "X Cup abcd1234", "mute": "Mute City"},
            log_enabled={slot_key: True, "mute": True},
        ),
        action_repeat=2,
        settings=DeficitBudgetSettings(
            uniform_fraction=0.7,
            min_weight=1.0,
            max_weight=3.0,
            ema_alpha=0.02,
            weight_update_rollouts=20,
            x_cup_generation_ema_alpha=0.1,
        ),
        restored_state=restored,
        seed=7,
    )

    entry = next(
        entry for entry in controller.runtime_state().entries if entry.course_key == slot_key
    )

    assert entry.generation_episode_count == 40
    assert entry.generation_finished_episode_count == 10
    assert entry.generation_success_sample_count == 40
    assert entry.generation_ema_completion_fraction == pytest.approx(0.686)
