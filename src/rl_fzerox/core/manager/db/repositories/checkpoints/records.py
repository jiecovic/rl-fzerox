# src/rl_fzerox/core/manager/db/repositories/checkpoints/records.py
"""Repository operations for installed published checkpoint rows."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from rl_fzerox.core.manager.db.models.checkpoints import PublishedCheckpointModel
from rl_fzerox.core.manager.db.repositories.checkpoints.mapping import (
    published_checkpoint_from_model,
)
from rl_fzerox.core.manager.models import ManagedPublishedCheckpoint


def insert_published_checkpoint(
    session: Session,
    *,
    checkpoint: ManagedPublishedCheckpoint,
    config_snapshot_id: str,
) -> ManagedPublishedCheckpoint:
    """Insert one installed checkpoint row."""

    if session.get(PublishedCheckpointModel, checkpoint.id) is not None:
        raise ValueError(f"published checkpoint {checkpoint.id!r} already exists")
    session.add(
        PublishedCheckpointModel(
            id=checkpoint.id,
            checkpoint_id=checkpoint.checkpoint_id,
            version=checkpoint.version,
            name=checkpoint.name,
            config_snapshot_id=config_snapshot_id,
            import_dir=str(checkpoint.import_dir),
            manifest_json=checkpoint.manifest_json,
            source_bundle_path=(
                None
                if checkpoint.source_bundle_path is None
                else str(checkpoint.source_bundle_path)
            ),
            source_bundle_sha256=checkpoint.source_bundle_sha256,
            source_run_id=checkpoint.source_run_id,
            source_run_name=checkpoint.source_run_name,
            source_artifact=checkpoint.source_artifact,
            local_num_timesteps=checkpoint.local_num_timesteps,
            lineage_num_timesteps=checkpoint.lineage_num_timesteps,
            policy_path=str(checkpoint.policy_path),
            model_path=str(checkpoint.model_path),
            checkpoint_metadata_path=str(checkpoint.checkpoint_metadata_path),
            train_config_path=str(checkpoint.train_config_path),
            evaluation_metrics_path=(
                None
                if checkpoint.evaluation_metrics_path is None
                else str(checkpoint.evaluation_metrics_path)
            ),
            engine_tuning_state_path=(
                None
                if checkpoint.engine_tuning_state_path is None
                else str(checkpoint.engine_tuning_state_path)
            ),
            engine_tuning_model_path=(
                None
                if checkpoint.engine_tuning_model_path is None
                else str(checkpoint.engine_tuning_model_path)
            ),
            exported_at=checkpoint.exported_at,
            imported_at=checkpoint.imported_at,
            updated_at=checkpoint.updated_at,
        )
    )
    session.flush()
    persisted = session.get(PublishedCheckpointModel, checkpoint.id)
    if persisted is None:
        raise RuntimeError(f"inserted checkpoint {checkpoint.id!r} could not be reloaded")
    return published_checkpoint_from_model(persisted)


def get_published_checkpoint(
    session: Session,
    checkpoint_id: str,
) -> ManagedPublishedCheckpoint | None:
    """Return one installed checkpoint by manager record id."""

    checkpoint = session.get(PublishedCheckpointModel, checkpoint_id)
    return None if checkpoint is None else published_checkpoint_from_model(checkpoint)


def list_published_checkpoints(session: Session) -> tuple[ManagedPublishedCheckpoint, ...]:
    """Return installed checkpoints in manager display order."""

    checkpoints = tuple(
        session.scalars(
            select(PublishedCheckpointModel).order_by(
                PublishedCheckpointModel.imported_at.desc(),
                PublishedCheckpointModel.id.desc(),
            )
        )
    )
    return tuple(published_checkpoint_from_model(checkpoint) for checkpoint in checkpoints)
