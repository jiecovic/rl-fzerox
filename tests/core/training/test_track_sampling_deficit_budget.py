# tests/core/training/test_track_sampling_deficit_budget.py
from pathlib import Path

import pytest

from rl_fzerox.apps.run_manager.api.payloads.track_sampling import (
    track_sampling_state_payload,
)
from rl_fzerox.core.domain.courses import generated_x_cup_slot_key
from rl_fzerox.core.training.session.callbacks.track_sampling import (
    DeficitBudgetCourseSchedulerState,
    DeficitBudgetSchedulerState,
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
    assert payload_entries["mute"]["completion_rate"] == pytest.approx(1.0)
    assert payload_entries["silence"]["completion_rate"] == pytest.approx(0.25)
    assert payload_entries["silence"]["ema_completion_fraction"] == pytest.approx(0.25)

    state_path = tmp_path / "track_sampling_state.json"
    save_track_sampling_runtime_state(state_path, runtime)
    restored = load_track_sampling_runtime_state(state_path)

    assert restored is not None
    assert restored.sampling_mode == "fixed_env"


def test_deficit_budget_runtime_state_roundtrips_scheduler_state(tmp_path: Path) -> None:
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
        deficit_budget_scheduler=DeficitBudgetSchedulerState(
            uniform_lane_deficit_steps=12.5,
            adaptive_lane_deficit_steps=-3.0,
            uniform_assignment_count=9,
            entries=(
                DeficitBudgetCourseSchedulerState(
                    course_key="easy",
                    uniform_deficit_steps=1.5,
                    adaptive_deficit_steps=0.0,
                    scheduler_env_steps=42,
                    last_uniform_assignment_index=8,
                ),
            ),
        ),
        entries=(
            TrackSamplingRuntimeEntry(
                track_id="easy",
                course_key="easy",
                label="Easy",
                base_weight=1.0,
                current_weight=1.0,
                completed_frames=80,
                episode_count=1,
                finished_episode_count=1,
                success_sample_count=1,
                ema_episode_frames=80.0,
                ema_completion_fraction=1.0,
            ),
        ),
    )
    state_path = tmp_path / "deficit_budget_state.json"

    save_track_sampling_runtime_state(state_path, state)
    restored = load_track_sampling_runtime_state(state_path)

    assert restored == state


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


def test_deficit_budget_payload_uses_scheduler_steps_for_step_share() -> None:
    state = TrackSamplingRuntimeState(
        sampling_mode="deficit_budget",
        action_repeat=2,
        update_episodes=20,
        ema_alpha=0.02,
        max_weight_scale=3.0,
        adaptive_completion_weight=0.5,
        adaptive_target_completion=1.0,
        adaptive_min_confidence_episodes=1,
        adaptive_confidence_scale=3.0,
        update_count=1,
        episodes_since_update=0,
        deficit_budget_scheduler=DeficitBudgetSchedulerState(
            entries=(
                DeficitBudgetCourseSchedulerState(course_key="easy", scheduler_env_steps=40),
                DeficitBudgetCourseSchedulerState(course_key="hard", scheduler_env_steps=60),
            ),
        ),
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
                current_weight=1.0,
                completed_frames=20,
                episode_count=1,
                finished_episode_count=0,
                success_sample_count=1,
                ema_episode_frames=20.0,
                ema_completion_fraction=0.1,
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

    assert entries["easy"]["completed_env_steps"] == 40
    assert entries["easy"]["measurement_env_steps"] == 0
    assert entries["easy"]["step_share"] == pytest.approx(0.4)
    assert entries["hard"]["completed_env_steps"] == 60
    assert entries["hard"]["measurement_env_steps"] == 10
    assert entries["hard"]["step_share"] == pytest.approx(0.6)


@pytest.mark.parametrize(
    ("focus_sharpness", "expected"),
    [
        (0.0, {"easy": 1 / 3, "medium": 1 / 3, "hard": 1 / 3}),
        (1.0, {"easy": 0.0, "medium": 1 / 3, "hard": 2 / 3}),
        (2.0, {"easy": 0.0, "medium": 0.2, "hard": 0.8}),
    ],
)
def test_deficit_budget_focus_uses_completion_gap_with_sharpness(
    focus_sharpness: float,
    expected: dict[str, float],
) -> None:
    controller = DeficitBudgetTrackSamplingController(
        resolved_courses=resolved_track_sampling_courses(
            {"easy": 1.0, "medium": 1.0, "hard": 1.0},
            course_keys={"easy": "easy", "medium": "medium", "hard": "hard"},
            log_keys={"easy": "easy", "medium": "medium", "hard": "hard"},
            labels={"easy": "Easy", "medium": "Medium", "hard": "Hard"},
            log_enabled={"easy": True, "medium": True, "hard": True},
        ),
        action_repeat=1,
        settings=DeficitBudgetSettings(
            uniform_fraction=0.0,
            focus_sharpness=focus_sharpness,
            ema_alpha=1.0,
            weight_update_rollouts=1,
            warmup_min_episodes_per_course=0,
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
                "track_id": "medium",
                "episode_step": 100,
                "episode_completion_fraction": 0.5,
                "termination_reason": "crashed",
            },
            {
                "track_id": "hard",
                "episode_step": 100,
                "episode_completion_fraction": 0.0,
                "termination_reason": "crashed",
            },
        )
    )
    controller.maybe_update_weights()
    values = _target_step_shares(controller)

    assert {course_key: values[course_key] for course_key in expected} == pytest.approx(expected)


def test_deficit_budget_warmup_keeps_targets_uniform_until_all_courses_have_samples() -> None:
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
            uniform_fraction=0.0,
            focus_sharpness=1.0,
            ema_alpha=1.0,
            weight_update_rollouts=1,
            warmup_min_episodes_per_course=2,
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
                "episode_completion_fraction": 0.0,
                "termination_reason": "crashed",
            },
        )
    )
    controller.maybe_update_weights()

    target_shares = _target_step_shares(controller)
    assert target_shares["easy"] == pytest.approx(0.5)
    assert target_shares["hard"] == pytest.approx(0.5)

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
                "episode_completion_fraction": 0.0,
                "termination_reason": "crashed",
            },
        )
    )
    controller.maybe_update_weights()

    target_shares = _target_step_shares(controller)
    assert target_shares["hard"] > 0.99
    assert target_shares["easy"] < 0.01


def test_deficit_budget_finish_metric_can_focus_failed_finish_with_high_completion() -> None:
    controller = DeficitBudgetTrackSamplingController(
        resolved_courses=resolved_track_sampling_courses(
            {"finished": 1.0, "retired": 1.0},
            course_keys={"finished": "finished", "retired": "retired"},
            log_keys={"finished": "finished", "retired": "retired"},
            labels={"finished": "Finished", "retired": "Retired"},
            log_enabled={"finished": True, "retired": True},
        ),
        action_repeat=1,
        settings=DeficitBudgetSettings(
            uniform_fraction=0.0,
            focus_sharpness=1.0,
            ema_alpha=1.0,
            weight_update_rollouts=1,
            difficulty_metric="finish_ema",
            warmup_min_episodes_per_course=0,
        ),
        seed=7,
    )

    controller.record_episodes(
        (
            {
                "track_id": "finished",
                "episode_step": 100,
                "episode_completion_fraction": 1.0,
                "termination_reason": "finished",
            },
            {
                "track_id": "retired",
                "episode_step": 100,
                "episode_completion_fraction": 1.0,
                "termination_reason": "retired",
            },
        )
    )
    controller.maybe_update_weights()
    entries = _runtime_entries(controller)
    target_shares = _target_step_shares(controller)

    assert entries["retired"].current_problem_score == pytest.approx(1.0)
    assert entries["finished"].current_problem_score == pytest.approx(0.0)
    assert target_shares["retired"] > 0.99
    assert target_shares["finished"] < 0.01


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
            focus_sharpness=1.0,
            ema_alpha=0.02,
            weight_update_rollouts=20,
        ),
        seed=7,
    )

    controller.add_rollout_budget(total_steps=200)
    assignments = tuple(
        _next_course_key(controller, fallback_assignment_steps=100.0) for _ in range(4)
    )

    assert assignments.count("mute") == 2
    assert assignments.count("silence") == 2


def test_deficit_budget_controller_can_drop_stale_queue_reservations() -> None:
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
            focus_sharpness=1.0,
            ema_alpha=0.02,
            weight_update_rollouts=20,
        ),
        seed=7,
    )

    controller.add_rollout_budget(total_steps=200)
    first_plan = controller.next_queued_reset(fallback_assignment_steps=100.0)
    controller.clear_reserved_assignments()
    replacement_plan = controller.next_queued_reset(fallback_assignment_steps=100.0)

    assert replacement_plan.course_id == first_plan.course_id


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
            focus_sharpness=1.0,
            ema_alpha=0.02,
            weight_update_rollouts=20,
        ),
        seed=7,
    )

    controller.add_rollout_budget(total_steps=200)
    refills = controller.refill_queues((0, 99), fallback_assignment_steps=100)

    assert set(refills) == {0}
    assert len(refills[0]) == DEFICIT_QUEUE_SETTINGS.minimum_refill_size
    refill_courses = tuple(queued_reset.course_id for queued_reset in refills[0])
    assert abs(refill_courses.count("mute") - refill_courses.count("silence")) <= 1


def test_deficit_budget_controller_keeps_uniform_and_adaptive_queue_lanes_separate() -> None:
    controller = DeficitBudgetTrackSamplingController(
        resolved_courses=resolved_track_sampling_courses(
            {"easy": 1.0, "medium": 1.0, "hard": 1.0},
            course_keys={"easy": "easy", "medium": "medium", "hard": "hard"},
            log_keys={"easy": "easy", "medium": "medium", "hard": "hard"},
            labels={"easy": "Easy", "medium": "Medium", "hard": "Hard"},
            log_enabled={"easy": True, "medium": True, "hard": True},
        ),
        action_repeat=1,
        settings=DeficitBudgetSettings(
            uniform_fraction=0.5,
            focus_sharpness=4.0,
            ema_alpha=1.0,
            weight_update_rollouts=1,
            warmup_min_episodes_per_course=0,
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
                "track_id": "medium",
                "episode_step": 100,
                "episode_completion_fraction": 1.0,
                "termination_reason": "finished",
            },
            {
                "track_id": "hard",
                "episode_step": 100,
                "episode_completion_fraction": 0.0,
                "termination_reason": "crashed",
            },
        )
    )
    controller.maybe_update_weights()
    controller.add_rollout_budget(total_steps=600)

    queued_resets = tuple(
        controller.next_queued_reset(fallback_assignment_steps=100.0) for _ in range(6)
    )
    uniform_courses = tuple(
        queued_reset.course_id
        for queued_reset in queued_resets
        if queued_reset.deficit_lane == "uniform"
    )
    adaptive_courses = tuple(
        queued_reset.course_id
        for queued_reset in queued_resets
        if queued_reset.deficit_lane == "adaptive"
    )

    assert set(uniform_courses) == {"easy", "medium", "hard"}
    assert adaptive_courses == ("hard", "hard", "hard")


def test_deficit_budget_controller_enforces_lane_budget_with_adaptive_backlog() -> None:
    restored = TrackSamplingRuntimeState(
        sampling_mode="deficit_budget",
        action_repeat=1,
        update_episodes=20,
        ema_alpha=1.0,
        max_weight_scale=4.0,
        adaptive_completion_weight=0.4,
        adaptive_target_completion=1.0,
        adaptive_min_confidence_episodes=1,
        adaptive_confidence_scale=4.0,
        update_count=1,
        episodes_since_update=0,
        entries=(
            TrackSamplingRuntimeEntry(
                track_id="easy",
                course_key="easy",
                label="Easy",
                base_weight=1.0,
                current_weight=1.0,
                completed_frames=10_000,
                episode_count=100,
                finished_episode_count=100,
                success_sample_count=100,
                ema_episode_frames=100.0,
                ema_completion_fraction=1.0,
            ),
            TrackSamplingRuntimeEntry(
                track_id="medium",
                course_key="medium",
                label="Medium",
                base_weight=1.0,
                current_weight=1.0,
                completed_frames=10_000,
                episode_count=100,
                finished_episode_count=100,
                success_sample_count=100,
                ema_episode_frames=100.0,
                ema_completion_fraction=1.0,
            ),
            TrackSamplingRuntimeEntry(
                track_id="hard",
                course_key="hard",
                label="Hard",
                base_weight=1.0,
                current_weight=1.0,
                completed_frames=0,
                episode_count=100,
                finished_episode_count=0,
                success_sample_count=100,
                ema_episode_frames=100.0,
                ema_completion_fraction=0.0,
            ),
        ),
    )
    controller = DeficitBudgetTrackSamplingController(
        resolved_courses=resolved_track_sampling_courses(
            {"easy": 1.0, "medium": 1.0, "hard": 1.0},
            course_keys={"easy": "easy", "medium": "medium", "hard": "hard"},
            log_keys={"easy": "easy", "medium": "medium", "hard": "hard"},
            labels={"easy": "Easy", "medium": "Medium", "hard": "Hard"},
            log_enabled={"easy": True, "medium": True, "hard": True},
        ),
        action_repeat=1,
        settings=DeficitBudgetSettings(
            uniform_fraction=0.6,
            focus_sharpness=4.0,
            ema_alpha=1.0,
            weight_update_rollouts=1,
            warmup_min_episodes_per_course=0,
        ),
        restored_state=restored,
        seed=7,
    )
    controller.add_rollout_budget(total_steps=1000)

    queued_resets = tuple(
        controller.next_queued_reset(fallback_assignment_steps=100.0) for _ in range(10)
    )
    lane_counts = {
        lane: sum(1 for queued_reset in queued_resets if queued_reset.deficit_lane == lane)
        for lane in ("uniform", "adaptive")
    }

    assert lane_counts == {"uniform": 6, "adaptive": 4}
    assert all(
        queued_reset.course_id == "hard"
        for queued_reset in queued_resets
        if queued_reset.deficit_lane == "adaptive"
    )


def test_deficit_budget_controller_staleness_guard_forces_uniform_coverage() -> None:
    controller = DeficitBudgetTrackSamplingController(
        resolved_courses=resolved_track_sampling_courses(
            {"easy": 1.0, "medium": 1.0, "stale": 1.0},
            course_keys={"easy": "easy", "medium": "medium", "stale": "stale"},
            log_keys={"easy": "easy", "medium": "medium", "stale": "stale"},
            labels={"easy": "Easy", "medium": "Medium", "stale": "Stale"},
            log_enabled={"easy": True, "medium": True, "stale": True},
        ),
        action_repeat=1,
        settings=DeficitBudgetSettings(
            uniform_fraction=1.0,
            focus_sharpness=1.0,
            ema_alpha=1.0,
            weight_update_rollouts=1,
            uniform_staleness_rotations=1.0,
            warmup_min_episodes_per_course=0,
        ),
        seed=7,
    )

    controller.add_rollout_budget(total_steps=900)
    controller.record_step_infos(({"track_id": "stale"},) * 500)

    first_assignments = tuple(
        controller.next_queued_reset(fallback_assignment_steps=100.0) for _ in range(3)
    )
    forced_assignment = controller.next_queued_reset(fallback_assignment_steps=100.0)

    assert all(queued_reset.course_id != "stale" for queued_reset in first_assignments)
    assert forced_assignment.course_id == "stale"
    assert forced_assignment.deficit_lane == "uniform"


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
            focus_sharpness=1.0,
            ema_alpha=0.02,
            weight_update_rollouts=20,
        ),
        seed=7,
    )

    controller.add_rollout_budget(total_steps=200)
    controller.record_step_infos(({"track_id": "mute"},) * 75)

    assert _next_course_key(controller, fallback_assignment_steps=1.0) == "silence"


def test_deficit_budget_controller_charges_steps_to_their_queue_lane() -> None:
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
            uniform_fraction=0.5,
            focus_sharpness=1.0,
            ema_alpha=0.02,
            weight_update_rollouts=20,
            warmup_min_episodes_per_course=0,
        ),
        seed=7,
    )

    controller.add_rollout_budget(total_steps=200)
    before = _scheduler_entries(controller)
    controller.record_step_infos(
        ({"track_id": "mute", "track_sampling_deficit_lane": "uniform"},) * 25
    )
    after = _scheduler_entries(controller)

    assert after["mute"].uniform_deficit_steps == before["mute"].uniform_deficit_steps - 25
    assert after["mute"].adaptive_deficit_steps == before["mute"].adaptive_deficit_steps


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
            focus_sharpness=1.0,
            ema_alpha=1.0,
            weight_update_rollouts=20,
            uniform_staleness_rotations=0.0,
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
    scheduler = runtime.deficit_budget_scheduler
    assert scheduler is not None
    scheduler_steps = {entry.course_key: entry.scheduler_env_steps for entry in scheduler.entries}
    assert scheduler_steps == {"mute": 40, "silence": 0}
    assert _next_course_key(controller, fallback_assignment_steps=1.0) == "silence"


def test_deficit_budget_controller_restores_exact_scheduler_debt() -> None:
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
        deficit_budget_scheduler=DeficitBudgetSchedulerState(
            uniform_lane_deficit_steps=250.0,
            adaptive_lane_deficit_steps=0.0,
            uniform_assignment_count=4,
            entries=(
                DeficitBudgetCourseSchedulerState(
                    course_key="balanced",
                    uniform_deficit_steps=-50.0,
                    scheduler_env_steps=1000,
                    last_uniform_assignment_index=4,
                ),
                DeficitBudgetCourseSchedulerState(
                    course_key="crash",
                    uniform_deficit_steps=300.0,
                    scheduler_env_steps=100,
                    last_uniform_assignment_index=1,
                ),
            ),
        ),
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
            focus_sharpness=1.0,
            ema_alpha=0.02,
            weight_update_rollouts=20,
        ),
        restored_state=restored,
        seed=7,
    )

    assert _next_course_key(controller, fallback_assignment_steps=100.0) == "crash"
    runtime = controller.runtime_state()
    assert runtime.deficit_budget_scheduler is not None
    restored_steps = {
        entry.course_key: entry.scheduler_env_steps
        for entry in runtime.deficit_budget_scheduler.entries
    }
    assert restored_steps == {"balanced": 1000, "crash": 100}


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
            focus_sharpness=1.0,
            ema_alpha=1.0,
            weight_update_rollouts=20,
            uniform_staleness_rotations=0.0,
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
        _next_course_key(controller, fallback_assignment_steps=100.0) for _ in range(5)
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
            focus_sharpness=1.0,
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
            focus_sharpness=1.0,
            ema_alpha=1.0,
            weight_update_rollouts=1,
            warmup_min_episodes_per_course=0,
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
    entries = _runtime_entries(controller)
    target_shares = _target_step_shares(controller)

    assert entries["hard"].current_problem_score > entries["easy"].current_problem_score
    assert target_shares["hard"] > target_shares["easy"]
    assert target_shares["hard"] < 0.7


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
            focus_sharpness=1.0,
            ema_alpha=0.02,
            weight_update_rollouts=20,
            warmup_min_episodes_per_course=0,
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
    target_shares = _target_step_shares(controller)
    assert target_shares["silence"] > target_shares["mute"]


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
            focus_sharpness=1.0,
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


def _target_step_shares(
    controller: DeficitBudgetTrackSamplingController,
) -> dict[str, float]:
    payload = track_sampling_state_payload(controller.runtime_state())
    raw_entries = payload["entries"]
    assert isinstance(raw_entries, list)
    shares: dict[str, float] = {}
    for raw_entry in raw_entries:
        assert isinstance(raw_entry, dict)
        course_key = raw_entry["course_key"]
        target_step_share = raw_entry["target_step_share"]
        assert isinstance(course_key, str)
        assert isinstance(target_step_share, float)
        shares[course_key] = target_step_share
    return shares


def _runtime_entries(
    controller: DeficitBudgetTrackSamplingController,
) -> dict[str, TrackSamplingRuntimeEntry]:
    return {entry.course_key: entry for entry in controller.runtime_state().entries}


def _next_course_key(
    controller: DeficitBudgetTrackSamplingController,
    *,
    fallback_assignment_steps: float,
) -> str:
    return controller.next_queued_reset(
        fallback_assignment_steps=fallback_assignment_steps,
    ).course_id


def _scheduler_entries(
    controller: DeficitBudgetTrackSamplingController,
) -> dict[str, DeficitBudgetCourseSchedulerState]:
    scheduler = controller.runtime_state().deficit_budget_scheduler
    assert scheduler is not None
    return {entry.course_key: entry for entry in scheduler.entries}
