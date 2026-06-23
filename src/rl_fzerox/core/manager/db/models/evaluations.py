# src/rl_fzerox/core/manager/db/models/evaluations.py
"""ORM models for manager-owned evaluation runs."""

from __future__ import annotations

from sqlalchemy import ForeignKey, Index, Integer, Text
from sqlalchemy.orm import Mapped, mapped_column

from rl_fzerox.core.manager.db.models.base import ManagerBase


class EvaluationModel(ManagerBase):
    """Persistent evaluation identity and immutable input snapshot."""

    __tablename__ = "evaluations"
    __table_args__ = (
        Index("evaluations_status_created_idx", "status", "created_at", "id"),
        Index("evaluations_source_run_idx", "source_run_id", "created_at", "id"),
    )

    id: Mapped[str] = mapped_column(primary_key=True)
    name: Mapped[str]
    status: Mapped[str]
    evaluation_dir: Mapped[str] = mapped_column(unique=True)
    source_run_id: Mapped[str | None] = mapped_column(ForeignKey("runs.id"))
    source_artifact: Mapped[str | None]
    policy_mode: Mapped[str]
    seed: Mapped[int] = mapped_column(Integer)
    target_json: Mapped[str] = mapped_column(Text)
    config_json: Mapped[str] = mapped_column(Text)
    checkpoint_json: Mapped[str] = mapped_column(Text)
    result_json_path: Mapped[str | None]
    error_message: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[str]
    updated_at: Mapped[str]
    started_at: Mapped[str | None]
    finished_at: Mapped[str | None]
