# src/rl_fzerox/core/manager/db/repositories/save_games/records.py
"""Repository operations for save-game identity rows."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from rl_fzerox.core.manager.db.models.save_games import (
    SaveGameAttemptModel,
    SaveGameCourseSetupModel,
    SaveGameCupSetupModel,
    SaveGameModel,
)
from rl_fzerox.core.manager.db.repositories.save_games.mapping import save_game_from_model
from rl_fzerox.core.manager.errors import ManagerNameConflictError
from rl_fzerox.core.manager.models import ManagedSaveGame, SaveGameStatus


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
            runner_device=save_game.runner_device,
            runner_renderer=save_game.runner_renderer,
            runner_policy_mode=save_game.runner_policy_mode,
            runner_attempt_seed=save_game.runner_attempt_seed,
            runner_recording_enabled=save_game.runner_recording_enabled,
            runner_recording_input_hud_enabled=save_game.runner_recording_input_hud_enabled,
            runner_recording_upscale_factor=save_game.runner_recording_upscale_factor,
            runner_recording_path=(
                None
                if save_game.runner_recording_path is None
                else str(save_game.runner_recording_path)
            ),
            runner_target_restart_on_retire=save_game.runner_target_restart_on_retire,
            runner_target_clear_goal=save_game.runner_target_clear_goal,
            runner_keep_failed_recordings=save_game.runner_keep_failed_recordings,
            runner_reload_policy_between_attempts=(save_game.runner_reload_policy_between_attempts),
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


def update_save_game_runner_settings(
    session: Session,
    *,
    save_game_id: str,
    device: Literal["cpu", "cuda"],
    renderer: Literal["angrylion", "gliden64"],
    policy_mode: Literal["deterministic", "stochastic"],
    attempt_seed: int | None,
    recording_enabled: bool,
    recording_input_hud_enabled: bool,
    recording_upscale_factor: int,
    recording_path: Path | None,
    target_restart_on_retire: bool,
    target_clear_goal: int,
    keep_failed_recordings: bool,
    reload_policy_between_attempts: bool,
    updated_at: str,
) -> ManagedSaveGame | None:
    """Update saved Career runner launch settings."""

    row = session.get(SaveGameModel, save_game_id)
    if row is None:
        return None
    row.runner_device = device
    row.runner_renderer = renderer
    row.runner_policy_mode = policy_mode
    row.runner_attempt_seed = attempt_seed
    row.runner_recording_enabled = recording_enabled
    row.runner_recording_input_hud_enabled = recording_input_hud_enabled
    row.runner_recording_upscale_factor = recording_upscale_factor
    row.runner_recording_path = None if recording_path is None else str(recording_path)
    row.runner_target_restart_on_retire = target_restart_on_retire
    row.runner_target_clear_goal = target_clear_goal
    row.runner_keep_failed_recordings = keep_failed_recordings
    row.runner_reload_policy_between_attempts = reload_policy_between_attempts
    row.updated_at = updated_at
    return save_game_from_model(row)


def delete_save_game(session: Session, save_game_id: str) -> ManagedSaveGame | None:
    """Delete one save-game identity row and its manager-owned child records."""

    row = session.get(SaveGameModel, save_game_id)
    if row is None:
        return None
    save_game = save_game_from_model(row)
    session.execute(
        delete(SaveGameAttemptModel).where(SaveGameAttemptModel.save_game_id == save_game_id)
    )
    session.execute(
        delete(SaveGameCourseSetupModel).where(
            SaveGameCourseSetupModel.save_game_id == save_game_id
        )
    )
    session.execute(
        delete(SaveGameCupSetupModel).where(SaveGameCupSetupModel.save_game_id == save_game_id)
    )
    session.delete(row)
    return save_game


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
