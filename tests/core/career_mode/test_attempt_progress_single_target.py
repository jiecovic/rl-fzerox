# tests/core/career_mode/test_attempt_progress_single_target.py
"""Tests for single-target Career Mode progress transitions."""

from __future__ import annotations

from pathlib import Path

from tests.core.career_mode.progress_helpers import (
    _race_setup,
    _Session,
    _SingleTargetCompletionStore,
)

from rl_fzerox.core.career_mode.progress.attempt import CareerAttemptProgress


def test_single_target_success_waits_for_post_gp_recording_boundary(tmp_path: Path) -> None:
    store = _SingleTargetCompletionStore(tmp_path)
    progress = CareerAttemptProgress(
        store=store,
        save_game_id=store.save_game.id,
        attempt_id="attempt-1",
        single_target=True,
        target_clear_goal=1,
    )

    terminal_transition = progress.handle_terminal_race(
        session=_Session(),
        setup=_race_setup(),
        info={"termination_reason": "finished", "position": 1, "race_time_ms": 88_333},
    )
    result_transition = progress.sync_post_terminal_progress(
        session=_Session(),
        setup=_race_setup(),
        info={"game_mode": "results", "termination_reason": "finished"},
    )
    post_gp_transition = progress.sync_post_terminal_progress(
        session=_Session(),
        setup=_race_setup(),
        info={"game_mode": "gp_end_cutscene", "termination_reason": "finished"},
    )
    credits_transition = progress.sync_post_terminal_progress(
        session=_Session(),
        setup=_race_setup(),
        info={"game_mode": "unskippable_credits", "termination_reason": "finished"},
    )

    assert terminal_transition.attempt_finished is False
    assert result_transition.attempt_finished is False
    assert post_gp_transition.attempt_finished is False
    assert credits_transition.attempt_finished is True
    assert credits_transition.next_plan is None
    assert credits_transition.finished_attempt_id == "attempt-1"
    assert credits_transition.finished_status == "succeeded"
    assert progress.attempt_id is None
    assert store.finished_attempts == [("attempt-1", "succeeded", None)]
    assert store.status_updates == ["paused"]
    assert store.started_next_attempt_count == 0


def test_single_target_success_finishes_on_unskippable_credits(tmp_path: Path) -> None:
    store = _SingleTargetCompletionStore(tmp_path)
    progress = CareerAttemptProgress(
        store=store,
        save_game_id=store.save_game.id,
        attempt_id="attempt-1",
        single_target=True,
        target_clear_goal=1,
    )

    terminal_transition = progress.handle_terminal_race(
        session=_Session(),
        setup=_race_setup(),
        info={"termination_reason": "finished", "position": 1, "race_time_ms": 88_333},
    )
    credits_transition = progress.sync_post_terminal_progress(
        session=_Session(),
        setup=_race_setup(),
        info={"game_mode": "unskippable_credits", "termination_reason": "finished"},
    )

    assert terminal_transition.attempt_finished is False
    assert credits_transition.attempt_finished is True
    assert credits_transition.next_plan is None
    assert credits_transition.finished_attempt_id == "attempt-1"
    assert credits_transition.finished_status == "succeeded"
    assert progress.attempt_id is None
    assert store.finished_attempts == [("attempt-1", "succeeded", None)]
    assert store.status_updates == ["paused"]
    assert store.started_next_attempt_count == 0


def test_single_target_explicit_post_gp_rank_two_counts_as_failed_attempt(
    tmp_path: Path,
) -> None:
    store = _SingleTargetCompletionStore(tmp_path)
    progress = CareerAttemptProgress(
        store=store,
        save_game_id=store.save_game.id,
        attempt_id="attempt-1",
        single_target=True,
        target_clear_goal=1,
    )

    terminal_transition = progress.handle_terminal_race(
        session=_Session(),
        setup=_race_setup(),
        info={"termination_reason": "finished", "position": 1, "race_time_ms": 88_333},
    )
    post_gp_transition = progress.sync_post_terminal_progress(
        session=_Session(),
        setup=_race_setup(),
        info={
            "game_mode": "gp_end_cutscene",
            "termination_reason": "finished",
            "career_mode_gp_final_rank": 2,
        },
    )

    assert terminal_transition.attempt_finished is False
    assert post_gp_transition.attempt_finished is True
    assert post_gp_transition.finished_attempt_id == "attempt-1"
    assert post_gp_transition.finished_status == "failed"
    assert post_gp_transition.finished_failure_reason == "gp cup final rank 2"
    assert post_gp_transition.next_plan is not None
    assert post_gp_transition.next_plan.attempt_id == "attempt-2"
    assert post_gp_transition.reset_emulator is True
    assert progress.attempt_id == "attempt-1"
    assert store.finished_attempts == [("attempt-1", "failed", "gp cup final rank 2")]
    assert store.status_updates == []
    assert store.started_next_attempt_count == 1


def test_single_target_result_screen_rank_two_counts_as_failed_attempt(
    tmp_path: Path,
) -> None:
    store = _SingleTargetCompletionStore(tmp_path)
    progress = CareerAttemptProgress(
        store=store,
        save_game_id=store.save_game.id,
        attempt_id="attempt-1",
        single_target=True,
        target_clear_goal=1,
    )

    progress.handle_terminal_race(
        session=_Session(),
        setup=_race_setup(),
        info={"termination_reason": "finished", "position": 1, "race_time_ms": 88_333},
    )
    post_gp_transition = progress.sync_post_terminal_progress(
        session=_Session(),
        setup=_race_setup(),
        info={
            "game_mode": "results",
            "termination_reason": "finished",
            "gp_final_rank": 2,
        },
    )
    assert post_gp_transition.attempt_finished is True
    assert post_gp_transition.finished_status == "failed"
    assert post_gp_transition.finished_failure_reason == "gp cup final rank 2"
    assert store.finished_attempts == [("attempt-1", "failed", "gp cup final rank 2")]


def test_single_target_post_gp_course_position_does_not_count_as_final_rank(
    tmp_path: Path,
) -> None:
    store = _SingleTargetCompletionStore(tmp_path)
    progress = CareerAttemptProgress(
        store=store,
        save_game_id=store.save_game.id,
        attempt_id="attempt-1",
        single_target=True,
        target_clear_goal=1,
    )

    progress.handle_terminal_race(
        session=_Session(),
        setup=_race_setup(),
        info={"termination_reason": "finished", "position": 1, "race_time_ms": 88_333},
    )
    post_gp_transition = progress.sync_post_terminal_progress(
        session=_Session(),
        setup=_race_setup(),
        info={"game_mode": "gp_end_cutscene", "termination_reason": "finished", "position": 2},
    )
    credits_transition = progress.sync_post_terminal_progress(
        session=_Session(),
        setup=_race_setup(),
        info={"game_mode": "unskippable_credits", "termination_reason": "finished", "position": 2},
    )

    assert post_gp_transition.attempt_finished is False
    assert credits_transition.attempt_finished is True
    assert credits_transition.finished_status == "succeeded"
    assert store.finished_attempts == [("attempt-1", "succeeded", None)]


def test_single_target_success_repeats_target_without_clear_goal(
    tmp_path: Path,
) -> None:
    store = _SingleTargetCompletionStore(tmp_path)
    progress = CareerAttemptProgress(
        store=store,
        save_game_id=store.save_game.id,
        attempt_id="attempt-1",
        single_target=True,
    )

    terminal_transition = progress.handle_terminal_race(
        session=_Session(),
        setup=_race_setup(),
        info={"termination_reason": "finished", "position": 1, "race_time_ms": 88_333},
    )
    post_gp_transition = progress.sync_post_terminal_progress(
        session=_Session(),
        setup=_race_setup(),
        info={"game_mode": "gp_end_cutscene", "termination_reason": "finished"},
    )
    credits_transition = progress.sync_post_terminal_progress(
        session=_Session(),
        setup=_race_setup(),
        info={"game_mode": "unskippable_credits", "termination_reason": "finished"},
    )

    assert terminal_transition.attempt_finished is False
    assert post_gp_transition.attempt_finished is False
    assert credits_transition.attempt_finished is True
    assert credits_transition.next_plan is not None
    assert credits_transition.next_plan.attempt_id == "attempt-2"
    assert credits_transition.finished_attempt_id == "attempt-1"
    assert credits_transition.finished_status == "succeeded"
    assert credits_transition.reset_emulator is True
    assert progress.attempt_id == "attempt-1"
    progress.apply_execution_plan(credits_transition.next_plan)
    assert progress.attempt_id == "attempt-2"
    assert store.finished_attempts == [("attempt-1", "succeeded", None)]
    assert store.status_updates == []
    assert store.started_next_attempt_count == 1


def test_single_target_success_stops_after_target_clear_goal(tmp_path: Path) -> None:
    store = _SingleTargetCompletionStore(tmp_path)
    progress = CareerAttemptProgress(
        store=store,
        save_game_id=store.save_game.id,
        attempt_id="attempt-1",
        single_target=True,
        target_clear_goal=2,
    )

    progress.handle_terminal_race(
        session=_Session(),
        setup=_race_setup(),
        info={"termination_reason": "finished", "position": 1, "race_time_ms": 88_333},
    )
    first_post_gp = progress.sync_post_terminal_progress(
        session=_Session(),
        setup=_race_setup(),
        info={"game_mode": "gp_end_cutscene", "termination_reason": "finished"},
    )
    first_clear = progress.sync_post_terminal_progress(
        session=_Session(),
        setup=_race_setup(),
        info={"game_mode": "unskippable_credits", "termination_reason": "finished"},
    )
    assert first_post_gp.attempt_finished is False
    assert first_clear.next_plan is not None
    progress.apply_execution_plan(first_clear.next_plan)

    progress.handle_terminal_race(
        session=_Session(),
        setup=_race_setup(),
        info={
            "termination_reason": "finished",
            "position": 1,
            "race_time_ms": 77_000,
            "course_index": 5,
        },
    )
    progress.observe_post_race_screen(
        info={
            "game_mode": "results",
            "termination_reason": "finished",
            "course_index": 5,
            "gp_final_rank": 1,
        },
        setup=_race_setup(),
    )
    second_post_gp = progress.sync_post_terminal_progress(
        session=_Session(),
        setup=_race_setup(),
        info={
            "game_mode": "gp_end_cutscene",
            "termination_reason": "finished",
            "gp_final_rank": 1,
        },
    )
    second_clear = progress.sync_post_terminal_progress(
        session=_Session(),
        setup=_race_setup(),
        info={
            "game_mode": "unskippable_credits",
            "termination_reason": "finished",
            "gp_final_rank": 1,
        },
    )

    assert first_clear.attempt_finished is True
    assert first_clear.finished_status == "succeeded"
    assert first_clear.next_plan.attempt_id == "attempt-2"
    assert second_post_gp.attempt_finished is False
    assert second_clear.attempt_finished is True
    assert second_clear.finished_status == "succeeded"
    assert second_clear.next_plan is None
    assert progress.attempt_id is None
    assert store.finished_attempts == [
        ("attempt-1", "succeeded", None),
        ("attempt-2", "succeeded", None),
    ]
    assert store.status_updates == ["paused"]
    assert store.started_next_attempt_count == 1


def test_single_target_success_finishes_after_return_to_main_menu(tmp_path: Path) -> None:
    store = _SingleTargetCompletionStore(tmp_path)
    progress = CareerAttemptProgress(
        store=store,
        save_game_id=store.save_game.id,
        attempt_id="attempt-1",
        single_target=True,
        target_clear_goal=1,
    )

    terminal_transition = progress.handle_terminal_race(
        session=_Session(),
        setup=_race_setup(),
        info={"termination_reason": "finished", "position": 1, "race_time_ms": 88_333},
    )
    menu_transition = progress.sync_post_terminal_progress(
        session=_Session(),
        setup=_race_setup(),
        info={"game_mode": "main_menu", "termination_reason": "finished"},
    )

    assert terminal_transition.attempt_finished is False
    assert menu_transition.attempt_finished is True
    assert menu_transition.next_plan is None
    assert menu_transition.finished_attempt_id == "attempt-1"
    assert menu_transition.finished_status == "succeeded"
    assert progress.attempt_id is None
    assert store.finished_attempts == [("attempt-1", "succeeded", None)]
    assert store.status_updates == ["paused"]
    assert store.started_next_attempt_count == 0


def test_single_target_success_does_not_finish_on_mid_cup_machine_settings(
    tmp_path: Path,
) -> None:
    store = _SingleTargetCompletionStore(tmp_path)
    progress = CareerAttemptProgress(
        store=store,
        save_game_id=store.save_game.id,
        attempt_id="attempt-1",
        single_target=True,
        target_clear_goal=1,
    )

    terminal_transition = progress.handle_terminal_race(
        session=_Session(),
        setup=_race_setup(),
        info={"termination_reason": "finished", "position": 1, "race_time_ms": 88_333},
    )
    menu_transition = progress.sync_post_terminal_progress(
        session=_Session(),
        setup=_race_setup(),
        info={"game_mode": "machine_settings", "termination_reason": "finished"},
    )

    assert terminal_transition.attempt_finished is False
    assert menu_transition.attempt_finished is False
    assert progress.attempt_id == "attempt-1"
    assert store.finished_attempts == []
    assert store.status_updates == []
    assert store.started_next_attempt_count == 0
