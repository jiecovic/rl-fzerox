# src/rl_fzerox/core/manager/db/repositories/save_games/attempts.py
"""Repository operations for save-game attempt rows."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from rl_fzerox.core.manager.db.models.save_games import SaveGameAttemptModel
from rl_fzerox.core.manager.db.repositories.save_games.mapping import save_attempt_from_model
from rl_fzerox.core.manager.models import ManagedSaveAttempt, SaveAttemptStatus


def insert_save_attempt(session: Session, attempt: ManagedSaveAttempt) -> None:
    """Insert one started unlock attempt."""

    session.add(
        SaveGameAttemptModel(
            id=attempt.id,
            save_game_id=attempt.save_game_id,
            status=attempt.status,
            target_kind=attempt.target_kind,
            difficulty=attempt.difficulty,
            cup_id=attempt.cup_id,
            course_id=attempt.course_id,
            started_at=attempt.started_at,
            finished_at=attempt.finished_at,
            finish_position=attempt.finish_position,
            finish_time_s=attempt.finish_time_s,
            failure_reason=attempt.failure_reason,
        )
    )


def get_save_attempt(session: Session, attempt_id: str) -> ManagedSaveAttempt | None:
    """Return one unlock attempt by id."""

    row = session.get(SaveGameAttemptModel, attempt_id)
    return None if row is None else save_attempt_from_model(row)


def list_save_attempts(
    session: Session,
    save_game_id: str,
) -> tuple[ManagedSaveAttempt, ...]:
    """Return attempts for one save game in newest-first order."""

    rows = session.scalars(
        select(SaveGameAttemptModel)
        .where(SaveGameAttemptModel.save_game_id == save_game_id)
        .order_by(SaveGameAttemptModel.started_at.desc(), SaveGameAttemptModel.id.desc())
    )
    return tuple(save_attempt_from_model(row) for row in rows)


def running_save_attempt(
    session: Session,
    save_game_id: str,
) -> ManagedSaveAttempt | None:
    """Return the active attempt for one save game, if any."""

    row = session.scalar(
        select(SaveGameAttemptModel)
        .where(
            SaveGameAttemptModel.save_game_id == save_game_id,
            SaveGameAttemptModel.status == "running",
        )
        .order_by(SaveGameAttemptModel.started_at.desc(), SaveGameAttemptModel.id.desc())
        .limit(1)
    )
    return None if row is None else save_attempt_from_model(row)


def fail_running_save_attempts(
    session: Session,
    *,
    save_game_id: str,
    finished_at: str,
    failure_reason: str,
) -> int:
    """Mark all orphaned running attempts for one save game as failed."""

    rows = session.scalars(
        select(SaveGameAttemptModel).where(
            SaveGameAttemptModel.save_game_id == save_game_id,
            SaveGameAttemptModel.status == "running",
        )
    )
    count = 0
    for row in rows:
        row.status = "failed"
        row.finished_at = finished_at
        row.failure_reason = failure_reason
        count += 1
    return count


def delete_running_save_attempts(
    session: Session,
    *,
    save_game_id: str,
) -> int:
    """Delete running attempts that should not become career history."""

    rows = session.scalars(
        select(SaveGameAttemptModel).where(
            SaveGameAttemptModel.save_game_id == save_game_id,
            SaveGameAttemptModel.status == "running",
        )
    )
    count = 0
    for row in rows:
        session.delete(row)
        count += 1
    return count


def finish_save_attempt(
    session: Session,
    *,
    attempt_id: str,
    status: SaveAttemptStatus,
    finished_at: str,
    finish_position: int | None = None,
    finish_time_s: float | None = None,
    failure_reason: str | None = None,
) -> ManagedSaveAttempt | None:
    """Mark one attempt terminal."""

    row = session.get(SaveGameAttemptModel, attempt_id)
    if row is None:
        return None
    row.status = status
    row.finished_at = finished_at
    row.finish_position = finish_position
    row.finish_time_s = finish_time_s
    row.failure_reason = failure_reason
    return save_attempt_from_model(row)
