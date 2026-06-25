# src/rl_fzerox/core/manager/registry/save_games/attempts.py
"""Save-game attempt lifecycle operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rl_fzerox.core.career_mode.progress.unlocks import build_unlock_progress
from rl_fzerox.core.manager.db.repositories import save_games as save_game_repository
from rl_fzerox.core.manager.models import (
    ManagedSaveAttempt,
    ManagedSaveUnlockProgress,
    ManagedSaveUnlockTarget,
    SaveAttemptStatus,
)
from rl_fzerox.core.manager.registry.common import new_record_id, utc_now
from rl_fzerox.core.manager.registry.save_games.execution import (
    find_unlock_progress_target,
    validate_policy_attempt_setup,
)

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

    from rl_fzerox.core.manager.store import ManagerStore


def start_save_attempt(
    store: ManagerStore,
    *,
    save_game_id: str,
    target_kind: str | None = None,
    difficulty: str | None = None,
    cup_id: str | None = None,
    course_id: str | None = None,
) -> ManagedSaveAttempt:
    """Record the start of one unlock attempt."""

    store.initialize()
    now = utc_now()
    attempt = ManagedSaveAttempt(
        id=new_record_id(f"{save_game_id} attempt"),
        save_game_id=save_game_id,
        target_kind=target_kind,
        status="running",
        difficulty=difficulty,
        cup_id=cup_id,
        course_id=course_id,
        started_at=now,
    )
    with store._orm_session() as session:
        if save_game_repository.get_save_game(session, save_game_id) is None:
            raise KeyError("save game not found")
        save_game_repository.insert_save_attempt(session, attempt)
        save_game_repository.touch_save_game(
            session,
            save_game_id=save_game_id,
            updated_at=now,
        )
    return attempt


def start_next_save_attempt(
    store: ManagerStore,
    save_game_id: str,
) -> ManagedSaveAttempt:
    """Resolve and record the next policy-backed unlock attempt."""

    store.initialize()
    started_at = utc_now()
    with store._orm_session() as session:
        save_game = save_game_repository.get_save_game(session, save_game_id)
        if save_game is None:
            raise KeyError("save game not found")
        attempts = save_game_repository.list_save_attempts(session, save_game_id)
        if any(attempt.status == "running" for attempt in attempts):
            raise ValueError("save game already has a running attempt")
        progress = build_unlock_progress(save_game.save_path, attempts=attempts)
        if progress.next_target is None:
            raise ValueError("save game has no pending unlock target")
        return _insert_policy_backed_save_attempt(
            session,
            save_game_id=save_game_id,
            progress=progress,
            target=progress.next_target,
            started_at=started_at,
            error_subject="next unlock target",
        )


def start_target_save_attempt(
    store: ManagerStore,
    save_game_id: str,
    *,
    target_kind: str,
    difficulty: str,
    cup_id: str,
    course_id: str | None = None,
) -> ManagedSaveAttempt:
    """Resolve and record one selected policy-backed unlock attempt."""

    store.initialize()
    started_at = utc_now()
    with store._orm_session() as session:
        save_game = save_game_repository.get_save_game(session, save_game_id)
        if save_game is None:
            raise KeyError("save game not found")
        attempts = save_game_repository.list_save_attempts(session, save_game_id)
        if any(attempt.status == "running" for attempt in attempts):
            raise ValueError("save game already has a running attempt")
        progress = build_unlock_progress(save_game.save_path, attempts=attempts)
        target = find_unlock_progress_target(
            progress,
            target_kind=target_kind,
            difficulty=difficulty,
            cup_id=cup_id,
            course_id=course_id,
        )
        if target is None:
            raise ValueError("selected unlock target is not part of the unlock path")
        if target.status not in {"pending", "succeeded"}:
            raise ValueError(f"selected unlock target is {target.status}")
        return _insert_policy_backed_save_attempt(
            session,
            save_game_id=save_game_id,
            progress=progress,
            target=target,
            started_at=started_at,
            error_subject="selected unlock target",
        )


def start_or_reuse_next_save_attempt(
    store: ManagerStore,
    save_game_id: str,
) -> ManagedSaveAttempt:
    """Return the active attempt, or start the next pending unlock target."""

    store.initialize()
    with store._orm_session() as session:
        if save_game_repository.get_save_game(session, save_game_id) is None:
            raise KeyError("save game not found")
        running_attempt = save_game_repository.running_save_attempt(session, save_game_id)
        if running_attempt is not None:
            return running_attempt
    return start_next_save_attempt(store, save_game_id)


def list_save_attempts(
    store: ManagerStore,
    save_game_id: str,
) -> tuple[ManagedSaveAttempt, ...]:
    """Return attempts for one save game."""

    store.initialize()
    with store._orm_session() as session:
        return save_game_repository.list_save_attempts(session, save_game_id)


def fail_running_save_attempts(
    store: ManagerStore,
    *,
    save_game_id: str,
    failure_reason: str,
) -> int:
    """Fail running attempts that are not backed by a live runner process."""

    store.initialize()
    finished_at = utc_now()
    with store._orm_session() as session:
        count = save_game_repository.fail_running_save_attempts(
            session,
            save_game_id=save_game_id,
            finished_at=finished_at,
            failure_reason=failure_reason,
        )
        if count:
            save_game_repository.touch_save_game(
                session,
                save_game_id=save_game_id,
                updated_at=finished_at,
            )
        return count


def discard_running_save_attempts(
    store: ManagerStore,
    *,
    save_game_id: str,
) -> int:
    """Delete attempts that never reached a real race result."""

    store.initialize()
    updated_at = utc_now()
    with store._orm_session() as session:
        count = save_game_repository.delete_running_save_attempts(
            session,
            save_game_id=save_game_id,
        )
        if count:
            save_game_repository.touch_save_game(
                session,
                save_game_id=save_game_id,
                updated_at=updated_at,
            )
        return count


def finish_save_attempt(
    store: ManagerStore,
    *,
    attempt_id: str,
    status: SaveAttemptStatus,
    finish_position: int | None = None,
    finish_time_s: float | None = None,
    failure_reason: str | None = None,
) -> ManagedSaveAttempt | None:
    """Mark one unlock attempt as succeeded or failed."""

    if status == "running":
        raise ValueError("finished attempts cannot keep running status")
    store.initialize()
    finished_at = utc_now()
    with store._orm_session() as session:
        attempt = save_game_repository.finish_save_attempt(
            session,
            attempt_id=attempt_id,
            status=status,
            finished_at=finished_at,
            finish_position=finish_position,
            finish_time_s=finish_time_s,
            failure_reason=failure_reason,
        )
        if attempt is not None:
            save_game_repository.touch_save_game(
                session,
                save_game_id=attempt.save_game_id,
                updated_at=finished_at,
            )
        return attempt


def _insert_policy_backed_save_attempt(
    session: Session,
    *,
    save_game_id: str,
    progress: ManagedSaveUnlockProgress,
    target: ManagedSaveUnlockTarget,
    started_at: str,
    error_subject: str,
) -> ManagedSaveAttempt:
    validate_policy_attempt_setup(
        session,
        save_game_id=save_game_id,
        progress=progress,
        target=target,
        timestamp=started_at,
        error_subject=error_subject,
    )
    attempt = ManagedSaveAttempt(
        id=new_record_id(f"{save_game_id} attempt"),
        save_game_id=save_game_id,
        target_kind=target.kind,
        status="running",
        difficulty=target.difficulty,
        cup_id=target.cup_id,
        course_id=target.course_id,
        started_at=started_at,
    )
    save_game_repository.insert_save_attempt(session, attempt)
    save_game_repository.touch_save_game(
        session,
        save_game_id=save_game_id,
        updated_at=started_at,
    )
    return attempt
