# src/rl_fzerox/core/manager/db/repositories/save_games/setups.py
"""Repository operations for save-game course and cup setup rows."""

from __future__ import annotations

from typing import Literal

from sqlalchemy import select
from sqlalchemy.orm import Session

from rl_fzerox.core.manager.db.models.save_games import (
    SaveGameCourseSetupModel,
    SaveGameCupSetupModel,
)
from rl_fzerox.core.manager.db.repositories.save_games.mapping import (
    course_setup_from_model,
    cup_setup_from_model,
)
from rl_fzerox.core.manager.models import ManagedSaveCourseSetup, ManagedSaveCupSetup


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
