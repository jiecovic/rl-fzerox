# src/rl_fzerox/core/manager/db/models/runs.py
"""ORM models for managed runs and their editable sources."""

from __future__ import annotations

from sqlalchemy import ForeignKey, Index, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from rl_fzerox.core.manager.db.models.base import ManagerBase
from rl_fzerox.core.manager.db.models.configs import ConfigSnapshotModel


class RunModel(ManagerBase):
    """Persistent identity and static metadata for one launched run."""

    __tablename__ = "runs"
    __table_args__ = (Index("runs_status_created_idx", "status", "created_at", "id"),)

    id: Mapped[str] = mapped_column(primary_key=True)
    name: Mapped[str]
    status: Mapped[str]
    config_snapshot_id: Mapped[str] = mapped_column(ForeignKey("config_snapshots.id"))
    run_dir: Mapped[str] = mapped_column(unique=True)
    lineage_id: Mapped[str | None]
    lineage_step_offset: Mapped[int]
    parent_run_id: Mapped[str | None] = mapped_column(ForeignKey("runs.id"))
    source_run_id: Mapped[str | None] = mapped_column(ForeignKey("runs.id"))
    source_artifact: Mapped[str | None]
    source_snapshot_dir: Mapped[str | None]
    source_num_timesteps: Mapped[int | None]
    created_at: Mapped[str]
    started_at: Mapped[str | None]
    stopped_at: Mapped[str | None]

    config_snapshot: Mapped[ConfigSnapshotModel] = relationship()


class RunDraftModel(ManagerBase):
    """Editable run configuration that can later be launched as a run."""

    __tablename__ = "run_drafts"
    __table_args__ = (Index("run_drafts_name_unique_idx", "name", unique=True),)

    id: Mapped[str] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(collation="NOCASE"))
    config_snapshot_id: Mapped[str] = mapped_column(ForeignKey("config_snapshots.id"))
    source_run_id: Mapped[str | None] = mapped_column(ForeignKey("runs.id"))
    source_artifact: Mapped[str | None]
    source_snapshot_dir: Mapped[str | None]
    source_num_timesteps: Mapped[int | None]
    created_at: Mapped[str]
    updated_at: Mapped[str]

    config_snapshot: Mapped[ConfigSnapshotModel] = relationship()


class RunTemplateModel(ManagerBase):
    """Reusable starting point for creating drafts."""

    __tablename__ = "run_templates"

    id: Mapped[str] = mapped_column(primary_key=True)
    name: Mapped[str]
    config_snapshot_id: Mapped[str] = mapped_column(ForeignKey("config_snapshots.id"))
    created_at: Mapped[str]
    updated_at: Mapped[str]

    config_snapshot: Mapped[ConfigSnapshotModel] = relationship()


class RunEventModel(ManagerBase):
    """Timestamped run event rendered in the run manager."""

    __tablename__ = "run_events"
    __table_args__ = (Index("run_events_run_id_created_idx", "run_id", "created_at", "id"),)

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id"))
    created_at: Mapped[str]
    kind: Mapped[str]
    message: Mapped[str]
