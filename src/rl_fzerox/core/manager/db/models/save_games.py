# src/rl_fzerox/core/manager/db/models/save_games.py
"""ORM models for portable save-game unlock state."""

from __future__ import annotations

from sqlalchemy import Float, ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from rl_fzerox.core.manager.db.models.base import ManagerBase


class SaveGameModel(ManagerBase):
    """Portable save-game identity and lifecycle state."""

    __tablename__ = "save_games"
    __table_args__ = (
        Index("save_games_created_idx", "created_at", "id"),
        Index("save_games_name_unique_idx", "name", unique=True),
    )

    id: Mapped[str] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(collation="NOCASE"))
    status: Mapped[str]
    save_path: Mapped[str] = mapped_column(unique=True)
    created_at: Mapped[str]
    updated_at: Mapped[str]
    last_finished_at: Mapped[str | None]


class SaveGameCourseSetupModel(ManagerBase):
    """Policy resolution rule for one save game."""

    __tablename__ = "save_game_course_setups"
    __table_args__ = (
        Index(
            "save_game_course_setups_save_scope_idx",
            "save_game_id",
            "scope",
            "difficulty",
            "cup_id",
            "course_id",
        ),
    )

    id: Mapped[str] = mapped_column(primary_key=True)
    save_game_id: Mapped[str] = mapped_column(ForeignKey("save_games.id"))
    scope: Mapped[str]
    difficulty: Mapped[str | None]
    cup_id: Mapped[str | None]
    course_id: Mapped[str | None]
    policy_run_id: Mapped[str] = mapped_column(ForeignKey("runs.id"))
    policy_artifact: Mapped[str]
    created_at: Mapped[str]
    updated_at: Mapped[str]


class SaveGameAttemptModel(ManagerBase):
    """One recorded policy attempt for one unlock-path target."""

    __tablename__ = "save_game_attempts"
    __table_args__ = (
        Index("save_game_attempts_save_started_idx", "save_game_id", "started_at", "id"),
        Index(
            "save_game_attempts_save_target_idx",
            "save_game_id",
            "target_kind",
            "difficulty",
            "cup_id",
            "course_id",
        ),
    )

    id: Mapped[str] = mapped_column(primary_key=True)
    save_game_id: Mapped[str] = mapped_column(ForeignKey("save_games.id"))
    policy_run_id: Mapped[str | None] = mapped_column(ForeignKey("runs.id"))
    policy_artifact: Mapped[str | None]
    status: Mapped[str]
    target_kind: Mapped[str | None]
    difficulty: Mapped[str | None]
    cup_id: Mapped[str | None]
    course_id: Mapped[str | None]
    started_at: Mapped[str]
    finished_at: Mapped[str | None]
    finish_position: Mapped[int | None] = mapped_column(Integer)
    finish_time_s: Mapped[float | None] = mapped_column(Float)
    failure_reason: Mapped[str | None] = mapped_column(Text)
