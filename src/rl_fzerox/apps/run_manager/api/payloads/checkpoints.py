# src/rl_fzerox/apps/run_manager/api/payloads/checkpoints.py
"""API payload builders for published checkpoint catalog and install state."""

from __future__ import annotations

from typing_extensions import TypedDict

from rl_fzerox.apps.run_manager.api.payloads.engine_tuning import engine_tuning_state_payload
from rl_fzerox.apps.run_manager.api.payloads.evaluations import (
    EvaluationResultSummaryPayload,
    evaluation_result_summary_payload_from_path,
)
from rl_fzerox.apps.run_manager.api.payloads.runs import run_summary_payload
from rl_fzerox.core.engine_tuning.config import engine_tuner_settings
from rl_fzerox.core.engine_tuning.persistence import load_engine_tuning_runtime_state
from rl_fzerox.core.manager import ManagedPublishedCheckpoint, ManagedRunSummary
from rl_fzerox.core.manager.checkpoints import CheckpointCatalog, CheckpointCatalogEntry
from rl_fzerox.core.manager.projection.engine_tuning import adaptive_engine_tuning_config


class PublishedCheckpointPayload(TypedDict):
    id: str
    checkpoint_id: str
    version: str
    name: str
    run_id: str
    run: dict[str, object] | None
    config: dict[str, object]
    import_dir: str
    source_run_id: str | None
    source_run_name: str | None
    source_artifact: str
    local_num_timesteps: int | None
    lineage_num_timesteps: int | None
    source_bundle_sha256: str | None
    has_evaluation_metrics: bool
    has_engine_tuning_state: bool
    evaluation_summary: EvaluationResultSummaryPayload | None
    engine_tuning_state: dict[str, object] | None
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
    snapshot_runs: dict[str, ManagedRunSummary] | None = None,
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
            published_checkpoint_payload(
                checkpoint,
                snapshot_run=(
                    None if snapshot_runs is None else snapshot_runs.get(checkpoint.run_id)
                ),
            )
            for checkpoint in installed_checkpoints
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
    *,
    snapshot_run: ManagedRunSummary | None = None,
) -> PublishedCheckpointPayload:
    run_payload = None
    if snapshot_run is not None:
        run_payload = run_summary_payload(snapshot_run)
        run_payload["available_policy_artifacts"] = list(_checkpoint_policy_artifacts(checkpoint))
    return {
        "id": checkpoint.id,
        "checkpoint_id": checkpoint.checkpoint_id,
        "version": checkpoint.version,
        "name": checkpoint.name,
        "run_id": checkpoint.run_id,
        "run": run_payload,
        "config": checkpoint.config.model_dump(mode="json"),
        "import_dir": str(checkpoint.import_dir),
        "source_run_id": checkpoint.source_run_id,
        "source_run_name": checkpoint.source_run_name,
        "source_artifact": checkpoint.source_artifact,
        "local_num_timesteps": checkpoint.local_num_timesteps,
        "lineage_num_timesteps": checkpoint.lineage_num_timesteps,
        "source_bundle_sha256": checkpoint.source_bundle_sha256,
        "has_evaluation_metrics": checkpoint.evaluation_metrics_path is not None,
        "has_engine_tuning_state": checkpoint.engine_tuning_state_path is not None,
        "evaluation_summary": evaluation_result_summary_payload_from_path(
            checkpoint.evaluation_metrics_path
        ),
        "engine_tuning_state": _checkpoint_engine_tuning_payload(checkpoint),
        "exported_at": checkpoint.exported_at,
        "imported_at": checkpoint.imported_at,
        "updated_at": checkpoint.updated_at,
    }


def _checkpoint_policy_artifacts(
    checkpoint: ManagedPublishedCheckpoint,
) -> tuple[str, ...]:
    if checkpoint.policy_path.is_file() and checkpoint.model_path.is_file():
        return (checkpoint.source_artifact,)
    return ()


def _checkpoint_engine_tuning_payload(
    checkpoint: ManagedPublishedCheckpoint,
) -> dict[str, object] | None:
    if (
        checkpoint.config.vehicle.engine_mode != "adaptive_tuner"
        or checkpoint.engine_tuning_state_path is None
    ):
        return None
    state = load_engine_tuning_runtime_state(
        checkpoint.engine_tuning_state_path,
        model_path=checkpoint.engine_tuning_model_path,
    )
    if state is None:
        return None
    return engine_tuning_state_payload(
        state,
        settings=engine_tuner_settings(adaptive_engine_tuning_config(checkpoint.config)),
    )
