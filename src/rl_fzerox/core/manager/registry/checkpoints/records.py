# src/rl_fzerox/core/manager/registry/checkpoints/records.py
"""Install and query published checkpoint bundle records."""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

from rl_fzerox.core.manager.checkpoints import (
    CheckpointBundleFileRole,
    CheckpointBundleManifest,
    import_checkpoint_bundle,
    read_checkpoint_bundle_manifest,
    serialize_checkpoint_bundle_manifest_json,
)
from rl_fzerox.core.manager.db.repositories import checkpoints as checkpoint_repository
from rl_fzerox.core.manager.db.repositories.configs import create_config_snapshot
from rl_fzerox.core.manager.models import ManagedPublishedCheckpoint
from rl_fzerox.core.manager.registry.common import slugify, utc_now
from rl_fzerox.core.manager.run_spec import ManagedRunConfig
from rl_fzerox.core.manager.storage.serialization import load_config_json

if TYPE_CHECKING:
    from rl_fzerox.core.manager.store import ManagerStore


def import_published_checkpoint_bundle(
    store: ManagerStore,
    *,
    bundle_path: Path,
    target_root: Path | None = None,
) -> ManagedPublishedCheckpoint:
    """Validate one checkpoint bundle, install files, and write its DB record."""

    store.initialize()
    manifest = read_checkpoint_bundle_manifest(bundle_path)
    record_id = _record_id(manifest)
    with store._orm_session() as session:
        if checkpoint_repository.get_published_checkpoint(session, record_id) is not None:
            raise ValueError(f"published checkpoint {record_id!r} already exists")

    import_result = import_checkpoint_bundle(
        bundle_path=bundle_path,
        target_root=target_root or store.checkpoints_root(),
        overwrite=False,
    )
    try:
        imported_at = utc_now()
        train_config_path = _role_path(import_result.import_dir, manifest, "train_config")
        config = load_config_json(train_config_path.read_text(encoding="utf-8"))
        source_bundle_path = bundle_path.expanduser().resolve()
        source_bundle_sha256 = _sha256_file(source_bundle_path)
        with store._orm_session() as session:
            config_snapshot = create_config_snapshot(
                session,
                kind="import",
                config=config,
                created_at=imported_at,
            )
            checkpoint = _checkpoint_from_import(
                manifest=manifest,
                record_id=record_id,
                import_dir=import_result.import_dir,
                source_bundle_path=source_bundle_path,
                source_bundle_sha256=source_bundle_sha256,
                imported_at=imported_at,
                config=config,
                config_hash=config_snapshot.config_hash,
            )
            return checkpoint_repository.insert_published_checkpoint(
                session,
                checkpoint=checkpoint,
                config_snapshot_id=config_snapshot.id,
            )
    except Exception:
        shutil.rmtree(import_result.import_dir, ignore_errors=True)
        raise


def get_published_checkpoint(
    store: ManagerStore,
    checkpoint_id: str,
) -> ManagedPublishedCheckpoint | None:
    """Return one installed checkpoint by manager record id."""

    store.initialize()
    with store._orm_session() as session:
        return checkpoint_repository.get_published_checkpoint(session, checkpoint_id)


def list_published_checkpoints(store: ManagerStore) -> tuple[ManagedPublishedCheckpoint, ...]:
    """Return installed published checkpoints in manager display order."""

    store.initialize()
    with store._orm_session() as session:
        return checkpoint_repository.list_published_checkpoints(session)


def _checkpoint_from_import(
    *,
    manifest: CheckpointBundleManifest,
    record_id: str,
    import_dir: Path,
    source_bundle_path: Path,
    source_bundle_sha256: str,
    imported_at: str,
    config: ManagedRunConfig,
    config_hash: str,
) -> ManagedPublishedCheckpoint:
    return ManagedPublishedCheckpoint(
        id=record_id,
        checkpoint_id=manifest.checkpoint.id,
        version=manifest.checkpoint.version,
        name=manifest.checkpoint.name,
        config=config,
        config_hash=config_hash,
        import_dir=import_dir,
        manifest_json=serialize_checkpoint_bundle_manifest_json(manifest),
        source_bundle_path=source_bundle_path,
        source_bundle_sha256=source_bundle_sha256,
        source_run_id=manifest.checkpoint.source_run_id,
        source_run_name=manifest.checkpoint.source_run_name,
        source_artifact=manifest.checkpoint.source_artifact,
        local_num_timesteps=manifest.checkpoint.local_num_timesteps,
        lineage_num_timesteps=manifest.checkpoint.lineage_num_timesteps,
        policy_path=_role_path(import_dir, manifest, "policy"),
        model_path=_role_path(import_dir, manifest, "model"),
        checkpoint_metadata_path=_role_path(import_dir, manifest, "checkpoint_metadata"),
        train_config_path=_role_path(import_dir, manifest, "train_config"),
        evaluation_metrics_path=_optional_role_path(import_dir, manifest, "evaluation_metrics"),
        engine_tuning_state_path=_optional_role_path(import_dir, manifest, "engine_tuning_state"),
        engine_tuning_model_path=_optional_role_path(import_dir, manifest, "engine_tuning_model"),
        exported_at=manifest.exported_at,
        imported_at=imported_at,
        updated_at=imported_at,
    )


def _record_id(manifest: CheckpointBundleManifest) -> str:
    checkpoint_slug = slugify(manifest.checkpoint.id)
    version_slug = slugify(manifest.checkpoint.version)
    if not checkpoint_slug or not version_slug:
        raise ValueError("checkpoint id and version must produce a manager record id")
    return f"{checkpoint_slug}-{version_slug}"


def _role_path(
    import_dir: Path,
    manifest: CheckpointBundleManifest,
    role: CheckpointBundleFileRole,
) -> Path:
    path = _optional_role_path(import_dir, manifest, role)
    if path is None:
        raise ValueError(f"checkpoint bundle is missing {role} payload")
    return path


def _optional_role_path(
    import_dir: Path,
    manifest: CheckpointBundleManifest,
    role: CheckpointBundleFileRole,
) -> Path | None:
    for bundle_file in manifest.files:
        if bundle_file.role == role:
            return import_dir.joinpath(*PurePosixPath(bundle_file.path).parts)
    return None


def _sha256_file(path: Path) -> str:
    sha256 = hashlib.sha256()
    with path.expanduser().resolve().open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            sha256.update(chunk)
    return sha256.hexdigest()
