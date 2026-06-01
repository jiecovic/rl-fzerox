# src/rl_fzerox/core/manager/db/models/metadata.py
"""ORM models for manager metadata and filesystem operation queues."""

from __future__ import annotations

from sqlalchemy.orm import Mapped, mapped_column

from rl_fzerox.core.manager.db.models.base import ManagerBase


class SchemaVersionModel(ManagerBase):
    """Single-row current schema marker."""

    __tablename__ = "schema_version"

    version: Mapped[int] = mapped_column(primary_key=True)
    applied_at: Mapped[str]


class FilesystemOperationModel(ManagerBase):
    """Deferred filesystem operation drained after DB transactions."""

    __tablename__ = "filesystem_operations"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    kind: Mapped[str]
    source_path: Mapped[str]
    target_path: Mapped[str | None]
    created_at: Mapped[str]


class LineageGroupModel(ManagerBase):
    """Display/TensorBoard grouping for one lineage."""

    __tablename__ = "lineage_groups"

    lineage_id: Mapped[str] = mapped_column(primary_key=True)
    group_name: Mapped[str] = mapped_column(primary_key=True)
    updated_at: Mapped[str]
