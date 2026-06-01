# src/rl_fzerox/core/manager/db/models/configs.py
"""ORM model for immutable managed-run configuration snapshots."""

from __future__ import annotations

from sqlalchemy.orm import Mapped, mapped_column

from rl_fzerox.core.manager.db.models.base import ManagerBase


class ConfigSnapshotModel(ManagerBase):
    """Canonical Pydantic-validated config captured at one manager state change."""

    __tablename__ = "config_snapshots"

    id: Mapped[str] = mapped_column(primary_key=True)
    kind: Mapped[str]
    schema_version: Mapped[int]
    created_at: Mapped[str]
    config_json: Mapped[str]
    config_hash: Mapped[str]
