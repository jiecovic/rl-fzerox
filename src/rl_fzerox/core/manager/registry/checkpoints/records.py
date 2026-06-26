# src/rl_fzerox/core/manager/registry/checkpoints/records.py
"""Install and query published checkpoint bundle records."""

from __future__ import annotations

import hashlib
import shutil
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from rl_fzerox.core.manager.artifacts.fork_source import link_or_copy_file
from rl_fzerox.core.manager.checkpoints import (
    CheckpointBundleFileRole,
    CheckpointBundleManifest,
    import_checkpoint_bundle,
    read_checkpoint_bundle_manifest,
    serialize_checkpoint_bundle_manifest_json,
)
from rl_fzerox.core.manager.db.models import (
    RunCommandModel,
    RunDraftModel,
    RunModel,
    SaveGameCourseSetupModel,
)
from rl_fzerox.core.manager.db.models.runtime import RunRuntimeModel
from rl_fzerox.core.manager.db.repositories import checkpoints as checkpoint_repository
from rl_fzerox.core.manager.db.repositories.configs import create_config_snapshot
from rl_fzerox.core.manager.db.repositories.filesystem import queue_delete_tree
from rl_fzerox.core.manager.db.repositories.runs import (
    append_run_event,
    get_managed_run,
    insert_run,
)
from rl_fzerox.core.manager.models import ManagedPublishedCheckpoint, ManagedRun
from rl_fzerox.core.manager.registry.common import slugify, utc_now
from rl_fzerox.core.manager.run_spec import ManagedRunConfig
from rl_fzerox.core.manager.storage.serialization import load_config_json
from rl_fzerox.core.training.runs import RUN_LAYOUT, save_train_run_config
from rl_fzerox.core.training.session.artifacts import (
    engine_tuning_checkpoint_path,
    engine_tuning_model_path,
    load_policy_artifact_metadata,
    policy_artifact_metadata_path,
)

if TYPE_CHECKING:
    from rl_fzerox.core.manager.store import ManagerStore


@dataclass(frozen=True, slots=True)
class _CheckpointSnapshotPaths:
    policy_path: Path
    model_path: Path
    metadata_path: Path
    train_config_path: Path
    engine_tuning_state_path: Path | None
    engine_tuning_model_path: Path | None


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
    run_id = _snapshot_run_id(record_id)
    with store._orm_session() as session:
        if checkpoint_repository.get_published_checkpoint(session, record_id) is not None:
            raise ValueError(f"published checkpoint {record_id!r} already exists")
        if get_managed_run(session, run_id) is not None:
            raise ValueError(f"checkpoint run snapshot {run_id!r} already exists")

    import_result = import_checkpoint_bundle(
        bundle_path=bundle_path,
        target_root=target_root or store.checkpoints_root(),
        overwrite=False,
    )
    try:
        imported_at = utc_now()
        train_config_path = _role_path(import_result.import_dir, manifest, "train_config")
        config = load_config_json(train_config_path.read_text(encoding="utf-8"))
        snapshot_paths = _materialize_run_snapshot(
            manifest=manifest,
            run_id=run_id,
            run_dir=import_result.import_dir,
            config=config,
        )
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
                run_id=run_id,
                import_dir=import_result.import_dir,
                source_bundle_path=source_bundle_path,
                source_bundle_sha256=source_bundle_sha256,
                imported_at=imported_at,
                config=config,
                config_hash=config_snapshot.config_hash,
                snapshot_paths=snapshot_paths,
            )
            _insert_or_repair_archived_run_snapshot(
                session,
                checkpoint=checkpoint,
                config_snapshot_id=config_snapshot.id,
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
    _repair_installed_checkpoint_snapshots(store, checkpoint_ids=(checkpoint_id,))
    with store._orm_session() as session:
        return checkpoint_repository.get_published_checkpoint(session, checkpoint_id)


def list_published_checkpoints(store: ManagerStore) -> tuple[ManagedPublishedCheckpoint, ...]:
    """Return installed published checkpoints in manager display order."""

    store.initialize()
    _repair_installed_checkpoint_snapshots(store)
    with store._orm_session() as session:
        return checkpoint_repository.list_published_checkpoints(session)


def delete_published_checkpoint(store: ManagerStore, checkpoint_id: str) -> bool:
    """Delete one installed checkpoint record and queue its imported files for cleanup."""

    store.initialize()
    deleted_at = utc_now()
    with store._orm_session() as session:
        checkpoint = checkpoint_repository.get_published_checkpoint(session, checkpoint_id)
        if checkpoint is None:
            return False
        if (
            session.scalar(
                select(SaveGameCourseSetupModel.id)
                .where(SaveGameCourseSetupModel.policy_source_kind == "checkpoint")
                .where(SaveGameCourseSetupModel.policy_source_id == checkpoint_id)
                .limit(1)
            )
            is not None
        ):
            raise ValueError("remove save-game course setups that still use this checkpoint")
        _assert_snapshot_run_can_be_deleted(session, run_id=checkpoint.run_id)
        deleted = checkpoint_repository.delete_published_checkpoint(session, checkpoint_id)
        if deleted is None:
            return False

    if store.get_run(checkpoint.run_id) is not None:
        store.delete_run(checkpoint.run_id)
    else:
        with store._orm_session() as session:
            queue_delete_tree(session, path=deleted.import_dir, created_at=deleted_at)
    store._drain_pending_filesystem_operations()
    return True


def _checkpoint_from_import(
    *,
    manifest: CheckpointBundleManifest,
    record_id: str,
    run_id: str,
    import_dir: Path,
    source_bundle_path: Path,
    source_bundle_sha256: str,
    imported_at: str,
    config: ManagedRunConfig,
    config_hash: str,
    snapshot_paths: _CheckpointSnapshotPaths,
) -> ManagedPublishedCheckpoint:
    return ManagedPublishedCheckpoint(
        id=record_id,
        checkpoint_id=manifest.checkpoint.id,
        version=manifest.checkpoint.version,
        name=manifest.checkpoint.name,
        config=config,
        config_hash=config_hash,
        run_id=run_id,
        import_dir=import_dir,
        manifest_json=serialize_checkpoint_bundle_manifest_json(manifest),
        source_bundle_path=source_bundle_path,
        source_bundle_sha256=source_bundle_sha256,
        source_run_id=manifest.checkpoint.source_run_id,
        source_run_name=manifest.checkpoint.source_run_name,
        source_artifact=manifest.checkpoint.source_artifact,
        local_num_timesteps=manifest.checkpoint.local_num_timesteps,
        lineage_num_timesteps=manifest.checkpoint.lineage_num_timesteps,
        policy_path=snapshot_paths.policy_path,
        model_path=snapshot_paths.model_path,
        checkpoint_metadata_path=snapshot_paths.metadata_path,
        train_config_path=snapshot_paths.train_config_path,
        evaluation_metrics_path=_optional_role_path(import_dir, manifest, "evaluation_metrics"),
        engine_tuning_state_path=snapshot_paths.engine_tuning_state_path,
        engine_tuning_model_path=snapshot_paths.engine_tuning_model_path,
        exported_at=manifest.exported_at,
        imported_at=imported_at,
        updated_at=imported_at,
    )


def _assert_snapshot_run_can_be_deleted(session: Session, *, run_id: str) -> None:
    run = session.get(RunModel, run_id)
    if run is None:
        return
    if run.status == "running":
        raise ValueError("stop the checkpoint run snapshot before deleting it")
    if session.get(RunCommandModel, run_id) is not None:
        raise ValueError("wait for the pending checkpoint run command to finish before deleting it")
    if (
        session.scalar(
            select(RunModel.id)
            .where(or_(RunModel.parent_run_id == run_id, RunModel.source_run_id == run_id))
            .limit(1)
        )
        is not None
    ):
        raise ValueError("delete runs forked from this checkpoint before deleting it")
    if (
        session.scalar(
            select(RunDraftModel.id).where(RunDraftModel.source_run_id == run_id).limit(1)
        )
        is not None
    ):
        raise ValueError("delete or retarget fork drafts that still depend on this checkpoint")
    if (
        session.scalar(
            select(SaveGameCourseSetupModel.id)
            .where(SaveGameCourseSetupModel.policy_source_kind == "run")
            .where(SaveGameCourseSetupModel.policy_source_id == run_id)
            .limit(1)
        )
        is not None
    ):
        raise ValueError("remove save-game course setups that still use this checkpoint run")


def _repair_installed_checkpoint_snapshots(
    store: ManagerStore,
    *,
    checkpoint_ids: tuple[str, ...] | None = None,
) -> None:
    with store._orm_session() as session:
        checkpoints = checkpoint_repository.list_published_checkpoints(session)
        if checkpoint_ids is not None:
            checkpoint_id_set = set(checkpoint_ids)
            checkpoints = tuple(
                checkpoint for checkpoint in checkpoints if checkpoint.id in checkpoint_id_set
            )
        for checkpoint in checkpoints:
            if get_managed_run(session, checkpoint.run_id) is not None:
                continue
            snapshot_paths = _materialize_run_snapshot(
                manifest=read_checkpoint_bundle_manifest_from_record(checkpoint),
                run_id=checkpoint.run_id,
                run_dir=checkpoint.import_dir,
                config=checkpoint.config,
            )
            config_snapshot = create_config_snapshot(
                session,
                kind="import",
                config=checkpoint.config,
                created_at=checkpoint.imported_at,
            )
            repaired_checkpoint = ManagedPublishedCheckpoint(
                id=checkpoint.id,
                checkpoint_id=checkpoint.checkpoint_id,
                version=checkpoint.version,
                name=checkpoint.name,
                config=checkpoint.config,
                config_hash=config_snapshot.config_hash,
                run_id=checkpoint.run_id,
                import_dir=checkpoint.import_dir,
                manifest_json=checkpoint.manifest_json,
                source_bundle_path=checkpoint.source_bundle_path,
                source_bundle_sha256=checkpoint.source_bundle_sha256,
                source_run_id=checkpoint.source_run_id,
                source_run_name=checkpoint.source_run_name,
                source_artifact=checkpoint.source_artifact,
                local_num_timesteps=checkpoint.local_num_timesteps,
                lineage_num_timesteps=checkpoint.lineage_num_timesteps,
                policy_path=snapshot_paths.policy_path,
                model_path=snapshot_paths.model_path,
                checkpoint_metadata_path=snapshot_paths.metadata_path,
                train_config_path=snapshot_paths.train_config_path,
                evaluation_metrics_path=checkpoint.evaluation_metrics_path,
                engine_tuning_state_path=snapshot_paths.engine_tuning_state_path,
                engine_tuning_model_path=snapshot_paths.engine_tuning_model_path,
                exported_at=checkpoint.exported_at,
                imported_at=checkpoint.imported_at,
                updated_at=utc_now(),
            )
            _insert_or_repair_archived_run_snapshot(
                session,
                checkpoint=repaired_checkpoint,
                config_snapshot_id=config_snapshot.id,
            )
            checkpoint_repository.update_published_checkpoint_snapshot(
                session,
                checkpoint=repaired_checkpoint,
                config_snapshot_id=config_snapshot.id,
            )


def read_checkpoint_bundle_manifest_from_record(
    checkpoint: ManagedPublishedCheckpoint,
) -> CheckpointBundleManifest:
    from rl_fzerox.core.manager.checkpoints import parse_checkpoint_bundle_manifest_json

    return parse_checkpoint_bundle_manifest_json(checkpoint.manifest_json)


def _materialize_run_snapshot(
    *,
    manifest: CheckpointBundleManifest,
    run_id: str,
    run_dir: Path,
    config: ManagedRunConfig,
) -> _CheckpointSnapshotPaths:
    from rl_fzerox.core.manager.projection.launches import build_managed_train_app_config

    resolved_run_dir = run_dir.expanduser().resolve()
    source_policy_path = _role_path(resolved_run_dir, manifest, "policy")
    source_model_path = _role_path(resolved_run_dir, manifest, "model")
    source_metadata_path = _role_path(resolved_run_dir, manifest, "checkpoint_metadata")
    _validate_checkpoint_metadata(manifest=manifest, policy_path=source_policy_path)
    source_engine_tuning_state_path = _optional_role_path(
        resolved_run_dir,
        manifest,
        "engine_tuning_state",
    )
    source_engine_tuning_model_path = _optional_role_path(
        resolved_run_dir,
        manifest,
        "engine_tuning_model",
    )

    train_config = build_managed_train_app_config(config, run_id=run_id, run_dir=resolved_run_dir)
    train_config_path = save_train_run_config(config=train_config, run_dir=resolved_run_dir)

    artifact_paths: dict[str, tuple[Path, Path]] = {
        "latest": (
            resolved_run_dir / RUN_LAYOUT.policy_artifacts.latest,
            resolved_run_dir / RUN_LAYOUT.model_artifacts.latest,
        ),
        "best": (
            resolved_run_dir / RUN_LAYOUT.policy_artifacts.best,
            resolved_run_dir / RUN_LAYOUT.model_artifacts.best,
        ),
        "final": (
            resolved_run_dir / RUN_LAYOUT.policy_artifacts.final,
            resolved_run_dir / RUN_LAYOUT.model_artifacts.final,
        ),
    }
    for target_policy_path, target_model_path in artifact_paths.values():
        _link_or_copy_if_distinct(source_policy_path, target_policy_path)
        _link_or_copy_if_distinct(source_model_path, target_model_path)
        _link_or_copy_if_distinct(
            source_metadata_path,
            policy_artifact_metadata_path(target_policy_path),
        )
        if source_engine_tuning_state_path is not None:
            _link_or_copy_if_distinct(
                source_engine_tuning_state_path,
                engine_tuning_checkpoint_path(target_policy_path),
            )
        if source_engine_tuning_model_path is not None:
            _link_or_copy_if_distinct(
                source_engine_tuning_model_path,
                engine_tuning_model_path(target_policy_path),
            )

    artifact_policy_path, artifact_model_path = artifact_paths[manifest.checkpoint.source_artifact]
    engine_tuning_state_path = engine_tuning_checkpoint_path(artifact_policy_path)
    engine_tuning_model_state_path = engine_tuning_model_path(artifact_policy_path)
    return _CheckpointSnapshotPaths(
        policy_path=artifact_policy_path,
        model_path=artifact_model_path,
        metadata_path=policy_artifact_metadata_path(artifact_policy_path),
        train_config_path=train_config_path,
        engine_tuning_state_path=(
            engine_tuning_state_path if engine_tuning_state_path.is_file() else None
        ),
        engine_tuning_model_path=(
            engine_tuning_model_state_path if engine_tuning_model_state_path.is_file() else None
        ),
    )


def _insert_or_repair_archived_run_snapshot(
    session: Session,
    *,
    checkpoint: ManagedPublishedCheckpoint,
    config_snapshot_id: str,
) -> None:
    existing = get_managed_run(session, checkpoint.run_id)
    if existing is not None:
        if existing.status != "archived" or existing.run_dir != checkpoint.import_dir:
            raise ValueError(
                "checkpoint run id conflicts with a non-checkpoint run "
                f"{checkpoint.run_id!r}"
            )
        _upsert_archived_run_runtime(session, checkpoint=checkpoint)
        return

    run = ManagedRun(
        id=checkpoint.run_id,
        name=checkpoint.name,
        status="archived",
        config=checkpoint.config,
        config_hash=checkpoint.config_hash,
        run_dir=checkpoint.import_dir,
        lineage_id=checkpoint.id,
        lineage_step_offset=_checkpoint_lineage_step_offset(checkpoint),
        source_num_timesteps=checkpoint.local_num_timesteps,
        created_at=_checkpoint_created_at(checkpoint),
        stopped_at=checkpoint.exported_at,
    )
    insert_run(session, run=run, config_snapshot_id=config_snapshot_id)
    session.flush()
    append_run_event(
        session,
        run_id=run.id,
        created_at=checkpoint.imported_at,
        kind="archived",
        message="published checkpoint installed as archived run snapshot",
    )
    _upsert_archived_run_runtime(session, checkpoint=checkpoint)


def _upsert_archived_run_runtime(
    session: Session,
    *,
    checkpoint: ManagedPublishedCheckpoint,
) -> None:
    local_steps = checkpoint.local_num_timesteps or 0
    runtime = session.get(RunRuntimeModel, checkpoint.run_id)
    if runtime is None:
        session.add(
            RunRuntimeModel(
                run_id=checkpoint.run_id,
                total_timesteps=local_steps,
                num_timesteps=local_steps,
                progress_fraction=1.0,
                updated_at=checkpoint.exported_at,
                fps=None,
                episode_reward_mean=None,
                episode_length_mean=None,
                approx_kl=None,
                entropy_loss=None,
                value_loss=None,
                policy_gradient_loss=None,
            )
        )
        return
    runtime.total_timesteps = local_steps
    runtime.num_timesteps = local_steps
    runtime.progress_fraction = 1.0
    runtime.updated_at = checkpoint.exported_at


def _checkpoint_lineage_step_offset(checkpoint: ManagedPublishedCheckpoint) -> int:
    local_steps = checkpoint.local_num_timesteps
    lineage_steps = checkpoint.lineage_num_timesteps
    if local_steps is None or lineage_steps is None:
        return 0
    return max(0, lineage_steps - local_steps)


def _checkpoint_created_at(checkpoint: ManagedPublishedCheckpoint) -> str:
    manifest = read_checkpoint_bundle_manifest_from_record(checkpoint)
    return manifest.checkpoint.created_at or checkpoint.exported_at


def _link_or_copy_if_distinct(source: Path, destination: Path) -> None:
    resolved_source = source.expanduser().resolve()
    resolved_destination = destination.expanduser().resolve()
    if resolved_source == resolved_destination:
        return
    link_or_copy_file(resolved_source, resolved_destination)


def _record_id(manifest: CheckpointBundleManifest) -> str:
    checkpoint_slug = slugify(manifest.checkpoint.id)
    version_slug = slugify(manifest.checkpoint.version)
    if not checkpoint_slug or not version_slug:
        raise ValueError("checkpoint id and version must produce a manager record id")
    return f"{checkpoint_slug}-{version_slug}"


def _snapshot_run_id(record_id: str) -> str:
    return f"checkpoint-{record_id}"


def _validate_checkpoint_metadata(
    *,
    manifest: CheckpointBundleManifest,
    policy_path: Path,
) -> None:
    metadata = load_policy_artifact_metadata(policy_path)
    if metadata is None or metadata.num_timesteps is None:
        raise ValueError(f"checkpoint policy metadata is missing num_timesteps: {policy_path}")
    expected_local_steps = manifest.checkpoint.local_num_timesteps
    if expected_local_steps is not None and metadata.num_timesteps != expected_local_steps:
        raise ValueError(
            "checkpoint policy metadata disagrees with bundle manifest "
            f"local steps: {metadata.num_timesteps} != {expected_local_steps}"
        )
    expected_lineage_steps = manifest.checkpoint.lineage_num_timesteps
    if (
        expected_lineage_steps is not None
        and metadata.lineage_num_timesteps is not None
        and metadata.lineage_num_timesteps != expected_lineage_steps
    ):
        raise ValueError(
            "checkpoint policy metadata disagrees with bundle manifest "
            f"lineage steps: {metadata.lineage_num_timesteps} != {expected_lineage_steps}"
        )


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
