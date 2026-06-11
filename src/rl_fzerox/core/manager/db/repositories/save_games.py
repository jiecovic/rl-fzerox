# src/rl_fzerox/core/manager/db/repositories/save_games.py
"""Repository operations for manager-owned portable save games."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from rl_fzerox.core.manager.db.models.runs import RunModel
from rl_fzerox.core.manager.db.models.save_games import (
    SaveGameAttemptModel,
    SaveGameCourseSetupModel,
    SaveGameCupSetupModel,
    SaveGameModel,
)
from rl_fzerox.core.manager.errors import ManagerNameConflictError
from rl_fzerox.core.manager.models import (
    ManagedSaveAttempt,
    ManagedSaveCourseSetup,
    ManagedSaveCupSetup,
    ManagedSaveGame,
    SaveAttemptStatus,
    SaveGameStatus,
)
from rl_fzerox.core.manager.registry.common import (
    optional_float,
    optional_int,
    optional_source_artifact,
    save_attempt_status,
    save_game_status,
)


def assert_save_game_name_available(
    session: Session,
    name: str,
    *,
    exclude_id: str | None = None,
) -> None:
    """Reject save-game names that would collide case-insensitively."""

    statement = select(SaveGameModel.id).where(func.lower(SaveGameModel.name) == name.lower())
    existing_id = session.scalar(statement.limit(1))
    if existing_id is not None and existing_id != exclude_id:
        raise ManagerNameConflictError(kind="save game", name=name)


def run_exists(session: Session, run_id: str) -> bool:
    """Return whether a policy run exists."""

    return session.get(RunModel, run_id) is not None


def insert_save_game(session: Session, save_game: ManagedSaveGame) -> None:
    """Insert one save-game identity row."""

    session.add(
        SaveGameModel(
            id=save_game.id,
            name=save_game.name,
            status=save_game.status,
            save_path=str(save_game.save_path),
            created_at=save_game.created_at,
            updated_at=save_game.updated_at,
            last_finished_at=save_game.last_finished_at,
        )
    )


def get_save_game(session: Session, save_game_id: str) -> ManagedSaveGame | None:
    """Return one save game by id."""

    row = session.get(SaveGameModel, save_game_id)
    return None if row is None else save_game_from_model(row)


def list_save_games(session: Session) -> tuple[ManagedSaveGame, ...]:
    """Return save games in newest-first manager order."""

    rows = session.scalars(
        select(SaveGameModel).order_by(SaveGameModel.created_at.desc(), SaveGameModel.id.desc())
    )
    return tuple(save_game_from_model(row) for row in rows)


def update_save_game_status(
    session: Session,
    *,
    save_game_id: str,
    status: SaveGameStatus,
    updated_at: str,
    last_finished_at: str | None = None,
) -> ManagedSaveGame | None:
    """Update lifecycle timestamps and status for one save game."""

    row = session.get(SaveGameModel, save_game_id)
    if row is None:
        return None
    row.status = status
    row.updated_at = updated_at
    if last_finished_at is not None:
        row.last_finished_at = last_finished_at
    return save_game_from_model(row)


def rename_save_game(
    session: Session,
    *,
    save_game_id: str,
    name: str,
    updated_at: str,
) -> ManagedSaveGame | None:
    """Rename one save-game identity row."""

    row = session.get(SaveGameModel, save_game_id)
    if row is None:
        return None
    assert_save_game_name_available(session, name, exclude_id=save_game_id)
    row.name = name
    row.updated_at = updated_at
    return save_game_from_model(row)


def touch_save_game(
    session: Session,
    *,
    save_game_id: str,
    updated_at: str,
) -> None:
    """Advance the save game's updated timestamp."""

    row = session.get(SaveGameModel, save_game_id)
    if row is not None:
        row.updated_at = updated_at


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


def list_course_setups(
    session: Session,
    save_game_id: str,
) -> tuple[ManagedSaveCourseSetup, ...]:
    """Return course setup rules for one save game."""

    rows = session.scalars(
        select(SaveGameCourseSetupModel)
        .where(SaveGameCourseSetupModel.save_game_id == save_game_id)
        .order_by(
            SaveGameCourseSetupModel.updated_at.desc(),
            SaveGameCourseSetupModel.id.desc(),
        )
    )
    return tuple(course_setup_from_model(row) for row in rows)


def upsert_course_setup(
    session: Session,
    *,
    setup_id: str,
    save_game_id: str,
    policy_run_id: str,
    policy_artifact: Literal["latest", "best"],
    engine_setting_raw_value: int,
    created_at: str,
    updated_at: str,
    difficulty: str | None = None,
    cup_id: str | None = None,
    course_id: str | None = None,
) -> ManagedSaveCourseSetup:
    """Create or replace one matching course setup rule."""

    row = session.scalar(
        select(SaveGameCourseSetupModel).where(
            SaveGameCourseSetupModel.save_game_id == save_game_id,
            SaveGameCourseSetupModel.difficulty.is_(difficulty)
            if difficulty is None
            else SaveGameCourseSetupModel.difficulty == difficulty,
            SaveGameCourseSetupModel.cup_id.is_(cup_id)
            if cup_id is None
            else SaveGameCourseSetupModel.cup_id == cup_id,
            SaveGameCourseSetupModel.course_id.is_(course_id)
            if course_id is None
            else SaveGameCourseSetupModel.course_id == course_id,
        )
    )
    if row is None:
        row = SaveGameCourseSetupModel(
            id=setup_id,
            save_game_id=save_game_id,
            difficulty=difficulty,
            cup_id=cup_id,
            course_id=course_id,
            policy_run_id=policy_run_id,
            policy_artifact=policy_artifact,
            engine_setting_raw_value=engine_setting_raw_value,
            created_at=created_at,
            updated_at=updated_at,
        )
        session.add(row)
    else:
        row.policy_run_id = policy_run_id
        row.policy_artifact = policy_artifact
        row.engine_setting_raw_value = engine_setting_raw_value
        row.updated_at = updated_at
    session.flush()
    return course_setup_from_model(row)


def list_cup_setups(
    session: Session,
    save_game_id: str,
) -> tuple[ManagedSaveCupSetup, ...]:
    """Return cup vehicle setup rules for one save game."""

    rows = session.scalars(
        select(SaveGameCupSetupModel)
        .where(SaveGameCupSetupModel.save_game_id == save_game_id)
        .order_by(
            SaveGameCupSetupModel.updated_at.desc(),
            SaveGameCupSetupModel.id.desc(),
        )
    )
    return tuple(cup_setup_from_model(row) for row in rows)


def upsert_cup_setup(
    session: Session,
    *,
    setup_id: str,
    save_game_id: str,
    cup_id: str,
    vehicle_id: str,
    created_at: str,
    updated_at: str,
    difficulty: str | None = None,
) -> ManagedSaveCupSetup:
    """Create or replace one cup vehicle setup."""

    row = session.scalar(
        select(SaveGameCupSetupModel).where(
            SaveGameCupSetupModel.save_game_id == save_game_id,
            SaveGameCupSetupModel.difficulty.is_(difficulty)
            if difficulty is None
            else SaveGameCupSetupModel.difficulty == difficulty,
            SaveGameCupSetupModel.cup_id == cup_id,
        )
    )
    if row is None:
        row = SaveGameCupSetupModel(
            id=setup_id,
            save_game_id=save_game_id,
            difficulty=difficulty,
            cup_id=cup_id,
            vehicle_id=vehicle_id,
            created_at=created_at,
            updated_at=updated_at,
        )
        session.add(row)
    else:
        row.vehicle_id = vehicle_id
        row.updated_at = updated_at
    session.flush()
    return cup_setup_from_model(row)


def save_game_from_model(row: SaveGameModel) -> ManagedSaveGame:
    """Convert one ORM row into a domain save-game record."""

    return ManagedSaveGame(
        id=row.id,
        name=row.name,
        status=save_game_status(row.status),
        save_path=Path(row.save_path).expanduser().resolve(),
        created_at=row.created_at,
        updated_at=row.updated_at,
        last_finished_at=row.last_finished_at,
    )


def course_setup_from_model(
    row: SaveGameCourseSetupModel,
) -> ManagedSaveCourseSetup:
    """Convert one ORM row into a domain course setup."""

    engine_setting_raw_value = optional_int(row.engine_setting_raw_value)
    return ManagedSaveCourseSetup(
        id=row.id,
        save_game_id=row.save_game_id,
        policy_run_id=row.policy_run_id,
        policy_artifact=_required_policy_artifact(row.policy_artifact),
        engine_setting_raw_value=50
        if engine_setting_raw_value is None
        else engine_setting_raw_value,
        created_at=row.created_at,
        updated_at=row.updated_at,
        difficulty=row.difficulty,
        cup_id=row.cup_id,
        course_id=row.course_id,
    )


def cup_setup_from_model(
    row: SaveGameCupSetupModel,
) -> ManagedSaveCupSetup:
    """Convert one ORM row into a domain cup setup."""

    return ManagedSaveCupSetup(
        id=row.id,
        save_game_id=row.save_game_id,
        cup_id=row.cup_id,
        vehicle_id=row.vehicle_id,
        created_at=row.created_at,
        updated_at=row.updated_at,
        difficulty=row.difficulty,
    )


def save_attempt_from_model(row: SaveGameAttemptModel) -> ManagedSaveAttempt:
    """Convert one ORM row into a domain attempt record."""

    return ManagedSaveAttempt(
        id=row.id,
        save_game_id=row.save_game_id,
        status=save_attempt_status(row.status),
        target_kind=row.target_kind,
        difficulty=row.difficulty,
        cup_id=row.cup_id,
        course_id=row.course_id,
        started_at=row.started_at,
        finished_at=row.finished_at,
        finish_position=optional_int(row.finish_position),
        finish_time_s=optional_float(row.finish_time_s),
        failure_reason=row.failure_reason,
    )


def _required_policy_artifact(value: object) -> Literal["latest", "best"]:
    artifact = optional_source_artifact(value)
    if artifact is None:
        raise ValueError("course setup is missing policy artifact")
    return artifact
