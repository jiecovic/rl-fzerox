# tests/ui/test_career_mode_worker.py
from __future__ import annotations

from pathlib import Path

from rl_fzerox.core.manager import ManagerStore
from rl_fzerox.ui.watch.runtime.career_mode.attempts import fail_running_attempts
from rl_fzerox.ui.watch.runtime.career_mode.loop.runtime import (
    should_observe_policy_transition,
)


def test_career_worker_marks_running_attempt_failed(tmp_path: Path) -> None:
    store = ManagerStore(tmp_path / "manager" / "runs.db")
    save_game = store.create_save_game(name="Career Save", save_games_root=tmp_path / "saves")
    attempt = store.start_save_attempt(
        save_game_id=save_game.id,
        target_kind="clear_gp_cup",
        difficulty="novice",
        cup_id="jack",
    )
    updated = store.update_save_game_status(save_game_id=save_game.id, status="running")
    assert updated is not None

    fail_running_attempts(
        store,
        save_game_id=save_game.id,
        failure_reason="career mode runner failed",
    )

    refreshed = store.get_save_game(save_game.id)
    attempts = store.list_save_attempts(save_game.id)
    assert refreshed is not None
    assert refreshed.status == "paused"
    assert len(attempts) == 1
    assert attempts[0].id == attempt.id
    assert attempts[0].status == "failed"
    assert attempts[0].failure_reason == "career mode runner failed"


def test_career_worker_waits_for_race_exit_before_policy_start_observation() -> None:
    assert (
        should_observe_policy_transition(
            policy_owns_control=True,
            active_policy_started=False,
            info={"game_mode": "gp_race"},
        )
        is False
    )
    assert (
        should_observe_policy_transition(
            policy_owns_control=True,
            active_policy_started=False,
            info={"game_mode": "gp_race", "termination_reason": "finished"},
        )
        is False
    )
    assert (
        should_observe_policy_transition(
            policy_owns_control=True,
            active_policy_started=False,
            info={"game_mode": "gp_race_next_course", "termination_reason": "crashed"},
        )
        is True
    )
    assert (
        should_observe_policy_transition(
            policy_owns_control=True,
            active_policy_started=True,
            info={"game_mode": "gp_race"},
        )
        is True
    )
    assert (
        should_observe_policy_transition(
            policy_owns_control=False,
            active_policy_started=True,
            info={"game_mode": "gp_race_next_course", "termination_reason": "crashed"},
        )
        is False
    )
