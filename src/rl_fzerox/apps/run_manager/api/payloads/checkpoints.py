# src/rl_fzerox/apps/run_manager/api/payloads/checkpoints.py
"""API payload builders for published checkpoint catalog and install state."""

from __future__ import annotations

from typing_extensions import TypedDict

from rl_fzerox.core.manager import ManagedPublishedCheckpoint
from rl_fzerox.core.manager.checkpoints import CheckpointCatalog, CheckpointCatalogEntry


class PublishedCheckpointPayload(TypedDict):
    id: str
    checkpoint_id: str
    version: str
    name: str
    source_artifact: str
    local_num_timesteps: int | None
    lineage_num_timesteps: int | None
    source_bundle_sha256: str | None
    has_evaluation_metrics: bool
    has_engine_tuning_state: bool
    exported_at: str
    imported_at: str
    updated_at: str


class CheckpointCatalogEntryPayload(TypedDict):
    id: str
    version: str
    name: str
    bundle: dict[str, object]
    manifest: dict[str, object]
    installed_checkpoint_id: str | None


class CheckpointCatalogPayload(TypedDict):
    catalog: dict[str, object]
    entries: list[CheckpointCatalogEntryPayload]
    installed_checkpoints: list[PublishedCheckpointPayload]


def checkpoint_catalog_payload(
    catalog: CheckpointCatalog,
    *,
    installed_checkpoints: tuple[ManagedPublishedCheckpoint, ...],
) -> CheckpointCatalogPayload:
    installed_by_key = {
        (checkpoint.checkpoint_id, checkpoint.version): checkpoint
        for checkpoint in installed_checkpoints
    }
    return {
        "catalog": {
            "format_name": catalog.format_name,
            "schema_version": catalog.schema_version,
            "updated_at": catalog.updated_at,
        },
        "entries": [
            checkpoint_catalog_entry_payload(
                entry,
                installed_checkpoint=installed_by_key.get((entry.id, entry.version)),
            )
            for entry in catalog.entries
        ],
        "installed_checkpoints": [
            published_checkpoint_payload(checkpoint) for checkpoint in installed_checkpoints
        ],
    }


def checkpoint_catalog_entry_payload(
    entry: CheckpointCatalogEntry,
    *,
    installed_checkpoint: ManagedPublishedCheckpoint | None = None,
) -> CheckpointCatalogEntryPayload:
    return {
        "id": entry.id,
        "version": entry.version,
        "name": entry.manifest.checkpoint.name,
        "bundle": entry.bundle.model_dump(mode="json"),
        "manifest": entry.manifest.model_dump(mode="json"),
        "installed_checkpoint_id": (
            None if installed_checkpoint is None else installed_checkpoint.id
        ),
    }


def published_checkpoint_payload(
    checkpoint: ManagedPublishedCheckpoint,
) -> PublishedCheckpointPayload:
    return {
        "id": checkpoint.id,
        "checkpoint_id": checkpoint.checkpoint_id,
        "version": checkpoint.version,
        "name": checkpoint.name,
        "source_artifact": checkpoint.source_artifact,
        "local_num_timesteps": checkpoint.local_num_timesteps,
        "lineage_num_timesteps": checkpoint.lineage_num_timesteps,
        "source_bundle_sha256": checkpoint.source_bundle_sha256,
        "has_evaluation_metrics": checkpoint.evaluation_metrics_path is not None,
        "has_engine_tuning_state": checkpoint.engine_tuning_state_path is not None,
        "exported_at": checkpoint.exported_at,
        "imported_at": checkpoint.imported_at,
        "updated_at": checkpoint.updated_at,
    }
