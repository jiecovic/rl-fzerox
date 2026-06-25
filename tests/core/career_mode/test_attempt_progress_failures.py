# tests/core/career_mode/test_attempt_progress_failures.py
"""Tests for Career Mode failed-attempt and retry transitions."""

from __future__ import annotations

from pathlib import Path

from tests.core.career_mode.progress_helpers import (
    _FailedRetryStore,
    _race_setup,
    _Session,
    _Store,
)

from rl_fzerox.core.career_mode.progress.attempt import CareerAttemptProgress


def test_crashed_race_keeps_current_gp_attempt(tmp_path: Path) -> None:
    store = _Store(tmp_path)
    progress = CareerAttemptProgress(
        store=store,
        save_game_id=store.save_game.id,
        attempt_id="attempt-1",
    )

    transition = progress.handle_terminal_race(
        session=_Session(),
        setup=_race_setup(),
        info={"termination_reason": "crashed", "position": 30, "race_time_ms": 88_333},
    )

    assert transition.attempt_finished is False
    assert transition.next_plan is None
    assert store.finished_attempts == []
    assert store.started_next_attempt_count == 0
    assert store.save_game.save_path.read_bytes() == b"save-ram"


def test_single_target_failed_gp_exit_starts_retry_attempt(tmp_path: Path) -> None:
    store = _FailedRetryStore(tmp_path)
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
        info={"termination_reason": "retired", "position": 30, "race_time_ms": 12_345},
    )
    menu_transition = progress.sync_post_terminal_progress(
        session=_Session(),
        setup=_race_setup(),
        info={"game_mode": "main_menu"},
    )

    assert terminal_transition.attempt_finished is False
    assert menu_transition.attempt_finished is True
    assert menu_transition.finished_attempt_id == "attempt-1"
    assert menu_transition.finished_status == "failed"
    assert menu_transition.finished_failure_reason == "gp attempt returned to menu before cup clear"
    assert menu_transition.next_plan is not None
    assert menu_transition.next_plan.attempt_id == "attempt-2"
    assert progress.attempt_id == "attempt-1"
    progress.apply_execution_plan(menu_transition.next_plan)
    assert progress.attempt_id == "attempt-2"
    assert store.finished_attempts == [
        ("attempt-1", "failed", "gp attempt returned to menu before cup clear")
    ]
    assert store.status_updates == []
    assert store.started_next_attempt_count == 1


def test_perfect_run_failure_starts_fresh_target_attempt_without_persisting_save(
    tmp_path: Path,
) -> None:
    store = _FailedRetryStore(tmp_path)
    progress = CareerAttemptProgress(
        store=store,
        save_game_id=store.save_game.id,
        attempt_id="attempt-1",
        single_target=True,
        perfect_run=True,
    )

    transition = progress.handle_terminal_race(
        session=_Session(),
        setup=_race_setup(),
        info={"termination_reason": "crashed", "position": 30, "race_time_ms": 12_345},
    )

    assert transition.attempt_finished is True
    assert transition.finished_attempt_id == "attempt-1"
    assert transition.finished_status == "failed"
    assert transition.finished_failure_reason == "perfect run reset after crashed"
    assert transition.next_plan is not None
    assert transition.next_plan.attempt_id == "attempt-2"
    assert transition.reset_emulator is True
    assert not store.save_game.save_path.exists()
    assert progress.attempt_id == "attempt-1"
    progress.apply_execution_plan(transition.next_plan)
    assert progress.attempt_id == "attempt-2"
    assert store.finished_attempts == [("attempt-1", "failed", "perfect run reset after crashed")]
    assert store.status_updates == []
    assert store.started_next_attempt_count == 1


def test_perfect_run_failure_retries_cup_target_when_setup_has_course_id(
    tmp_path: Path,
) -> None:
    store = _FailedRetryStore(tmp_path)
    progress = CareerAttemptProgress(
        store=store,
        save_game_id=store.save_game.id,
        attempt_id="attempt-1",
        single_target=True,
        perfect_run=True,
    )

    transition = progress.handle_terminal_race(
        session=_Session(),
        setup=_race_setup(course_id="mute_city"),
        info={"termination_reason": "crashed", "position": 30, "race_time_ms": 12_345},
    )

    assert transition.attempt_finished is True
    assert transition.next_plan is not None
    assert store.next_attempt is not None
    assert store.next_attempt.target_kind == "clear_gp_cup"
    assert store.next_attempt.difficulty == "expert"
    assert store.next_attempt.cup_id == "jack"
    assert store.next_attempt.course_id is None
