# src/rl_fzerox/core/manager/db/models/checkpoints.py
"""ORM model for installed published checkpoint records."""

from __future__ import annotations

from sqlalchemy import ForeignKey, Index, Integer, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from rl_fzerox.core.manager.db.models.base import ManagerBase
from rl_fzerox.core.manager.db.models.configs import ConfigSnapshotModel


class PublishedCheckpointModel(ManagerBase):
    """Manager-owned record for one locally installed checkpoint release."""

    __tablename__ = "published_checkpoints"
    __table_args__ = (
        UniqueConstraint(
            "checkpoint_id",
            "version",
            name="published_checkpoints_identity_uq",
        ),
        Index("published_checkpoints_imported_idx", "imported_at", "id"),
    )

    id: Mapped[str] = mapped_column(primary_key=True)
    checkpoint_id: Mapped[str]
    version: Mapped[str]
    name: Mapped[str]
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id"), unique=True)
    config_snapshot_id: Mapped[str] = mapped_column(ForeignKey("config_snapshots.id"))
    import_dir: Mapped[str] = mapped_column(unique=True)
    manifest_json: Mapped[str] = mapped_column(Text)
    source_bundle_path: Mapped[str | None]
    source_bundle_sha256: Mapped[str | None]
    source_run_id: Mapped[str | None]
    source_run_name: Mapped[str | None]
    source_artifact: Mapped[str]
    local_num_timesteps: Mapped[int | None] = mapped_column(Integer)
    lineage_num_timesteps: Mapped[int | None] = mapped_column(Integer)
    policy_path: Mapped[str]
    model_path: Mapped[str]
    checkpoint_metadata_path: Mapped[str]
    train_config_path: Mapped[str]
    evaluation_metrics_path: Mapped[str | None]
    engine_tuning_state_path: Mapped[str | None]
    engine_tuning_model_path: Mapped[str | None]
    exported_at: Mapped[str]
    imported_at: Mapped[str]
    updated_at: Mapped[str]

    config_snapshot: Mapped[ConfigSnapshotModel] = relationship()
