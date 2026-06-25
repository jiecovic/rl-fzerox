# tests/core/career_mode/test_attempt_progress_replay.py
"""Tests for replaying already-cleared Career Mode targets."""

from __future__ import annotations

from pathlib import Path

from tests.core.career_mode.progress_helpers import (
    _race_setup,
    _ReplayTargetStore,
    _Session,
)

from rl_fzerox.core.career_mode.progress.attempt import CareerAttemptProgress


def test_replayed_target_retire_does_not_finish_attempt(tmp_path: Path) -> None:
    store = _ReplayTargetStore(tmp_path)
    progress = CareerAttemptProgress(
        store=store,
        save_game_id=store.save_game.id,
        attempt_id="attempt-1",
        single_target=True,
        target_clear_goal=1,
    )

    transition = progress.handle_terminal_race(
        session=_Session(),
        setup=_race_setup(),
        info={"termination_reason": "retired", "position": 30, "race_time_ms": 12_345},
    )

    assert transition.attempt_finished is False
    assert progress.attempt_id == "attempt-1"
    assert store.finished_attempts == []
    assert store.status_updates == []
    assert store.started_next_attempt_count == 0


def test_replayed_target_success_waits_for_post_gp_recording_boundary(tmp_path: Path) -> None:
    store = _ReplayTargetStore(tmp_path)
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
        info={
            "termination_reason": "finished",
            "position": 1,
            "race_time_ms": 88_333,
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
    result_transition = progress.sync_post_terminal_progress(
        session=_Session(),
        setup=_race_setup(),
        info={"game_mode": "results", "termination_reason": "finished"},
    )
    post_gp_transition = progress.sync_post_terminal_progress(
        session=_Session(),
        setup=_race_setup(),
        info={
            "game_mode": "gp_end_cutscene",
            "termination_reason": "finished",
            "gp_final_rank": 1,
        },
    )
    credits_transition = progress.sync_post_terminal_progress(
        session=_Session(),
        setup=_race_setup(),
        info={
            "game_mode": "unskippable_credits",
            "termination_reason": "finished",
            "gp_final_rank": 1,
        },
    )

    assert terminal_transition.attempt_finished is False
    assert result_transition.attempt_finished is False
    assert post_gp_transition.attempt_finished is False
    assert credits_transition.attempt_finished is True
    assert credits_transition.finished_status == "succeeded"
    assert progress.attempt_id is None
    assert store.finished_attempts == [("attempt-1", "succeeded", None)]
    assert store.status_updates == ["paused"]
    assert store.started_next_attempt_count == 0


def test_replayed_target_success_counts_post_gp_without_final_course_metadata(
    tmp_path: Path,
) -> None:
    store = _ReplayTargetStore(tmp_path)
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
    progress.observe_post_race_screen(
        info={
            "game_mode": "results",
            "termination_reason": "finished",
            "course_index": 5,
            "gp_final_rank": 1,
        },
        setup=_race_setup(),
    )
    credits_transition = progress.sync_post_terminal_progress(
        session=_Session(),
        setup=_race_setup(),
        info={
            "game_mode": "unskippable_credits",
            "termination_reason": "finished",
            "gp_final_rank": 1,
        },
    )

    assert terminal_transition.attempt_finished is False
    assert credits_transition.attempt_finished is True
    assert credits_transition.next_plan is None
    assert credits_transition.finished_status == "succeeded"
    assert progress.attempt_id is None
    assert store.finished_attempts == [("attempt-1", "succeeded", None)]
    assert store.status_updates == ["paused"]
    assert store.started_next_attempt_count == 0


def test_replayed_target_success_counts_when_ceremony_returns_to_menu(
    tmp_path: Path,
) -> None:
    store = _ReplayTargetStore(tmp_path)
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
    progress.observe_post_race_screen(
        info={
            "game_mode": "results",
            "termination_reason": "finished",
            "course_index": 5,
            "gp_final_rank": 1,
        },
        setup=_race_setup(),
    )
    post_gp_transition = progress.sync_post_terminal_progress(
        session=_Session(),
        setup=_race_setup(),
        info={
            "game_mode": "gp_end_cutscene",
            "termination_reason": "finished",
            "gp_final_rank": 1,
        },
    )
    menu_transition = progress.sync_post_terminal_progress(
        session=_Session(),
        setup=_race_setup(),
        info={"game_mode": "main_menu", "termination_reason": "finished"},
    )

    assert terminal_transition.attempt_finished is False
    assert post_gp_transition.attempt_finished is False
    assert menu_transition.attempt_finished is True
    assert menu_transition.next_plan is None
    assert menu_transition.finished_status == "succeeded"
    assert progress.attempt_id is None
    assert store.finished_attempts == [("attempt-1", "succeeded", None)]
    assert store.status_updates == ["paused"]
    assert store.started_next_attempt_count == 0


def test_replayed_target_missing_rank_at_ceremony_exit_fails_attempt(
    tmp_path: Path,
) -> None:
    store = _ReplayTargetStore(tmp_path)
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
        info={
            "termination_reason": "finished",
            "position": 1,
            "race_time_ms": 88_333,
            "course_index": 5,
        },
    )
    post_gp_transition = progress.sync_post_terminal_progress(
        session=_Session(),
        setup=_race_setup(),
        info={"game_mode": "gp_end_cutscene", "termination_reason": "finished"},
    )
    menu_transition = progress.sync_post_terminal_progress(
        session=_Session(),
        setup=_race_setup(),
        info={"game_mode": "main_menu", "termination_reason": "finished"},
    )

    assert terminal_transition.attempt_finished is False
    assert post_gp_transition.attempt_finished is False
    assert menu_transition.attempt_finished is True
    assert menu_transition.finished_status == "failed"
    assert menu_transition.finished_failure_reason == "gp cup final rank missing"
    assert menu_transition.next_plan is not None
    assert menu_transition.next_plan.attempt_id == "attempt-2"
    assert progress.attempt_id == "attempt-1"
    assert store.finished_attempts == [("attempt-1", "failed", "gp cup final rank missing")]
    assert store.status_updates == []
    assert store.started_next_attempt_count == 1


def test_replayed_target_ignores_stale_final_rank_from_race_terminal_frame(
    tmp_path: Path,
) -> None:
    store = _ReplayTargetStore(tmp_path)
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
        info={
            "game_mode": "gp_race",
            "termination_reason": "finished",
            "position": 1,
            "race_time_ms": 88_333,
            "course_index": 5,
            "gp_final_rank": 1,
        },
    )
    menu_transition = progress.sync_post_terminal_progress(
        session=_Session(),
        setup=_race_setup(),
        info={"game_mode": "main_menu", "termination_reason": "finished"},
    )

    assert terminal_transition.attempt_finished is False
    assert menu_transition.attempt_finished is True
    assert menu_transition.finished_status == "failed"
    assert menu_transition.finished_failure_reason == "gp cup final rank missing"
    assert store.finished_attempts == [("attempt-1", "failed", "gp cup final rank missing")]


def test_replayed_target_uses_final_rank_observed_before_ceremony(
    tmp_path: Path,
) -> None:
    store = _ReplayTargetStore(tmp_path)
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
    progress.observe_post_race_screen(
        info={
            "game_mode": "results",
            "termination_reason": "finished",
            "course_index": 5,
            "gp_final_rank": 1,
        },
        setup=_race_setup(),
    )
    post_gp_transition = progress.sync_post_terminal_progress(
        session=_Session(),
        setup=_race_setup(),
        info={"game_mode": "gp_end_cutscene", "termination_reason": "finished"},
    )
    menu_transition = progress.sync_post_terminal_progress(
        session=_Session(),
        setup=_race_setup(),
        info={"game_mode": "main_menu", "termination_reason": "finished"},
    )

    assert terminal_transition.attempt_finished is False
    assert post_gp_transition.attempt_finished is False
    assert menu_transition.attempt_finished is True
    assert menu_transition.finished_status == "succeeded"
    assert progress.attempt_id is None
    assert store.finished_attempts == [("attempt-1", "succeeded", None)]
    assert store.status_updates == ["paused"]
    assert store.started_next_attempt_count == 0


def test_replayed_target_uses_losing_rank_observed_before_ceremony(
    tmp_path: Path,
) -> None:
    store = _ReplayTargetStore(tmp_path)
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
    progress.observe_post_race_screen(
        info={
            "game_mode": "results",
            "termination_reason": "finished",
            "course_index": 5,
            "gp_final_rank": 2,
            "gp_points": 570,
        },
        setup=_race_setup(),
    )
    menu_transition = progress.sync_post_terminal_progress(
        session=_Session(),
        setup=_race_setup(),
        info={"game_mode": "main_menu", "termination_reason": "finished"},
    )

    assert terminal_transition.attempt_finished is False
    assert menu_transition.attempt_finished is True
    assert menu_transition.finished_status == "failed"
    assert menu_transition.finished_failure_reason == "gp cup final rank 2"
    assert menu_transition.recording_info is not None
    assert menu_transition.recording_info["career_mode_gp_final_rank"] == 2
    assert menu_transition.recording_info["career_mode_gp_points"] == 570
    assert store.finished_attempts == [("attempt-1", "failed", "gp cup final rank 2")]


def test_replayed_target_success_does_not_finish_after_nonfinal_course(
    tmp_path: Path,
) -> None:
    store = _ReplayTargetStore(tmp_path)
    progress = CareerAttemptProgress(
        store=store,
        save_game_id=store.save_game.id,
        attempt_id="attempt-1",
        single_target=True,
    )

    terminal_transition = progress.handle_terminal_race(
        session=_Session(),
        setup=_race_setup(),
        info={
            "termination_reason": "finished",
            "position": 1,
            "race_time_ms": 88_333,
            "course_index": 0,
        },
    )
    menu_transition = progress.sync_post_terminal_progress(
        session=_Session(),
        setup=_race_setup(),
        info={"game_mode": "main_menu", "termination_reason": "finished"},
    )

    assert terminal_transition.attempt_finished is False
    assert menu_transition.attempt_finished is False
    assert progress.attempt_id == "attempt-1"
    assert store.finished_attempts == []
    assert store.status_updates == []
    assert store.started_next_attempt_count == 0


def test_replayed_target_final_course_without_gp_win_does_not_count_as_clear(
    tmp_path: Path,
) -> None:
    store = _ReplayTargetStore(tmp_path)
    progress = CareerAttemptProgress(
        store=store,
        save_game_id=store.save_game.id,
        attempt_id="attempt-1",
        single_target=True,
    )

    terminal_transition = progress.handle_terminal_race(
        session=_Session(),
        setup=_race_setup(),
        info={
            "termination_reason": "finished",
            "position": 1,
            "race_time_ms": 88_333,
            "course_index": 5,
        },
    )
    menu_transition = progress.sync_post_terminal_progress(
        session=_Session(),
        setup=_race_setup(),
        info={"game_mode": "main_menu", "termination_reason": "finished"},
    )

    assert terminal_transition.attempt_finished is False
    assert menu_transition.attempt_finished is True
    assert menu_transition.finished_status == "failed"
    assert menu_transition.finished_failure_reason == "gp cup final rank missing"
    assert progress.attempt_id == "attempt-1"
    assert store.finished_attempts == [("attempt-1", "failed", "gp cup final rank missing")]
    assert store.status_updates == []
    assert store.started_next_attempt_count == 1
