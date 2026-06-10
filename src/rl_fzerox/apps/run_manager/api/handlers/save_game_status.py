# src/rl_fzerox/apps/run_manager/api/handlers/save_game_status.py
from __future__ import annotations

from rl_fzerox.apps.run_manager.api.payloads.save_games import save_game_payload
from rl_fzerox.apps.run_manager.launching.save_games import active_career_mode_runner_pid
from rl_fzerox.core.manager import ManagedSaveAttempt, ManagedSaveGame, ManagerStore
from rl_fzerox.core.manager.models import SaveGameStatus

_ORPHAN_RUNNER_FAILURE_REASON = "career mode runner process disappeared"


def save_game_payload_for_store(
    store: ManagerStore,
    save_game: ManagedSaveGame,
    *,
    cleanup_orphan_runner: bool = True,
) -> dict[str, object]:
    runner_active = _career_runner_active(store, save_game)
    unlock_progress = store.save_game_unlock_progress(save_game.id)
    attempts = store.list_save_attempts(save_game.id)
    if (
        cleanup_orphan_runner
        and any(attempt.status == "running" for attempt in attempts)
        and not runner_active
    ):
        store.fail_running_save_attempts(
            save_game_id=save_game.id,
            failure_reason=_ORPHAN_RUNNER_FAILURE_REASON,
        )
        updated = store.update_save_game_status(
            save_game_id=save_game.id,
            status=_status_after_orphan_runner_cleanup(
                attempts=attempts,
                completed_targets=unlock_progress.completed_count,
            ),
        )
        if updated is not None:
            save_game = updated
        unlock_progress = store.save_game_unlock_progress(save_game.id)
        attempts = store.list_save_attempts(save_game.id)
    if _should_reset_stale_unstarted_status(
        save_game=save_game,
        runner_active=runner_active,
        attempts=attempts,
        completed_targets=unlock_progress.completed_count,
    ):
        updated = store.update_save_game_status(save_game_id=save_game.id, status="created")
        if updated is not None:
            save_game = updated
    return save_game_payload(
        save_game,
        runner_active=runner_active,
        unlock_progress=unlock_progress,
        attempts=attempts,
        course_setups=store.list_save_course_setups(save_game.id),
    )


def _status_after_orphan_runner_cleanup(
    *,
    attempts: tuple[ManagedSaveAttempt, ...],
    completed_targets: int,
) -> SaveGameStatus:
    has_real_attempt = any(_real_terminal_race_attempt(attempt) for attempt in attempts)
    if completed_targets == 0 and not has_real_attempt:
        return "created"
    return "paused"


def _should_reset_stale_unstarted_status(
    *,
    save_game: ManagedSaveGame,
    runner_active: bool,
    attempts: tuple[ManagedSaveAttempt, ...],
    completed_targets: int,
) -> bool:
    if runner_active or save_game.status not in {"running", "paused"}:
        return False
    return completed_targets == 0 and not any(
        _real_terminal_race_attempt(attempt) for attempt in attempts
    )


def _real_terminal_race_attempt(attempt: ManagedSaveAttempt) -> bool:
    if attempt.status == "running" or attempt.finished_at is None:
        return False
    if attempt.status == "succeeded":
        return True
    return attempt.failure_reason in {"crashed", "depleted", "retired"}


def _career_runner_active(store: ManagerStore, save_game: ManagedSaveGame) -> bool:
    lease_id = store.viewer_lease_id(kind="career_mode", owner_id=save_game.id)
    return (
        active_career_mode_runner_pid(
            store=store,
            lease_id=lease_id,
            save_game_id=save_game.id,
        )
        is not None
    )
