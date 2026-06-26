# src/rl_fzerox/core/manager/db/repositories/checkpoints/mapping.py
"""ORM-to-domain mapping for installed checkpoint rows."""

from __future__ import annotations

from pathlib import Path

from rl_fzerox.core.manager.db.models.checkpoints import PublishedCheckpointModel
from rl_fzerox.core.manager.models import ManagedPublishedCheckpoint, PolicySourceArtifact
from rl_fzerox.core.manager.storage.serialization import load_config_json


def published_checkpoint_from_model(
    checkpoint: PublishedCheckpointModel,
) -> ManagedPublishedCheckpoint:
    """Build the public checkpoint dataclass from an ORM row."""

    return ManagedPublishedCheckpoint(
        id=checkpoint.id,
        checkpoint_id=checkpoint.checkpoint_id,
        version=checkpoint.version,
        name=checkpoint.name,
        config=load_config_json(checkpoint.config_snapshot.config_json),
        config_hash=checkpoint.config_snapshot.config_hash,
        import_dir=Path(checkpoint.import_dir),
        manifest_json=checkpoint.manifest_json,
        source_bundle_path=(
            None if checkpoint.source_bundle_path is None else Path(checkpoint.source_bundle_path)
        ),
        source_bundle_sha256=checkpoint.source_bundle_sha256,
        source_run_id=checkpoint.source_run_id,
        source_run_name=checkpoint.source_run_name,
        source_artifact=_source_artifact(checkpoint.source_artifact),
        local_num_timesteps=checkpoint.local_num_timesteps,
        lineage_num_timesteps=checkpoint.lineage_num_timesteps,
        policy_path=Path(checkpoint.policy_path),
        model_path=Path(checkpoint.model_path),
        checkpoint_metadata_path=Path(checkpoint.checkpoint_metadata_path),
        train_config_path=Path(checkpoint.train_config_path),
        evaluation_metrics_path=(
            None
            if checkpoint.evaluation_metrics_path is None
            else Path(checkpoint.evaluation_metrics_path)
        ),
        engine_tuning_state_path=(
            None
            if checkpoint.engine_tuning_state_path is None
            else Path(checkpoint.engine_tuning_state_path)
        ),
        engine_tuning_model_path=(
            None
            if checkpoint.engine_tuning_model_path is None
            else Path(checkpoint.engine_tuning_model_path)
        ),
        exported_at=checkpoint.exported_at,
        imported_at=checkpoint.imported_at,
        updated_at=checkpoint.updated_at,
    )


def _source_artifact(value: object) -> PolicySourceArtifact:
    match str(value):
        case "latest":
            return "latest"
        case "best":
            return "best"
        case "final":
            return "final"
    raise ValueError(f"unsupported checkpoint source artifact: {value!r}")
