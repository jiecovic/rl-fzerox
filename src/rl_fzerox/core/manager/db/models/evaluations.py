# src/rl_fzerox/core/manager/db/models/evaluations.py
"""ORM models for manager-owned evaluation runs."""

from __future__ import annotations

from sqlalchemy import ForeignKey, Index, Integer, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from rl_fzerox.core.manager.db.models.base import ManagerBase


class EvaluationModel(ManagerBase):
    """Persistent evaluation identity and immutable input snapshot."""

    __tablename__ = "evaluations"
    __table_args__ = (
        Index("evaluations_status_created_idx", "status", "created_at", "id"),
        Index("evaluations_source_run_idx", "source_run_id", "created_at", "id"),
        Index(
            "evaluations_source_policy_idx",
            "source_policy_kind",
            "source_policy_id",
            "created_at",
            "id",
        ),
    )

    id: Mapped[str] = mapped_column(primary_key=True)
    name: Mapped[str]
    status: Mapped[str]
    evaluation_dir: Mapped[str] = mapped_column(unique=True)
    source_policy_kind: Mapped[str]
    source_policy_id: Mapped[str | None]
    source_run_id: Mapped[str | None] = mapped_column(ForeignKey("runs.id"))
    source_artifact: Mapped[str | None]
    preset_id: Mapped[str] = mapped_column(ForeignKey("evaluation_presets.id"), nullable=False)
    preset_version: Mapped[int] = mapped_column(Integer, nullable=False)
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


class EvaluationPresetModel(ManagerBase):
    """SQLite-owned evaluation preset and target configuration."""

    __tablename__ = "evaluation_presets"

    id: Mapped[str] = mapped_column(primary_key=True)
    name: Mapped[str]
    version: Mapped[int] = mapped_column(Integer)
    seed: Mapped[int] = mapped_column(Integer)
    renderer: Mapped[str]
    target_json: Mapped[str] = mapped_column(Text)
    builtin: Mapped[bool]
    created_at: Mapped[str]
    updated_at: Mapped[str]


class EvaluationBaselineSuiteModel(ManagerBase):
    """Materialized reset-state suite owned by one preset version."""

    __tablename__ = "evaluation_baseline_suites"
    __table_args__ = (
        UniqueConstraint(
            "preset_id",
            "preset_version",
            name="evaluation_baseline_suites_preset_version_uq",
        ),
    )

    id: Mapped[str] = mapped_column(primary_key=True)
    preset_id: Mapped[str] = mapped_column(ForeignKey("evaluation_presets.id"))
    preset_version: Mapped[int] = mapped_column(Integer)
    status: Mapped[str]
    suite_dir: Mapped[str]
    manifest_path: Mapped[str | None]
    error_message: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[str]
    updated_at: Mapped[str]
    materialized_at: Mapped[str | None]
