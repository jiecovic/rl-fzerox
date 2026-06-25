# src/rl_fzerox/core/manager/transfer/archive.py
"""Import and export of portable managed run bundles."""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

from sqlalchemy.exc import IntegrityError as SqlAlchemyIntegrityError
from sqlalchemy.orm import Session

from rl_fzerox.core.manager.db.models import (
    LineageGroupModel,
    RunModel,
    RunRuntimeModel,
)
from rl_fzerox.core.manager.db.repositories.configs import create_config_snapshot
from rl_fzerox.core.manager.db.repositories.runs import append_run_event, insert_run
from rl_fzerox.core.manager.models import ManagedRun, ManagedRunRuntime, RunStatus
from rl_fzerox.core.manager.run_spec import ManagedRunConfig
from rl_fzerox.core.manager.store import ManagerStore
from rl_fzerox.core.manager.transfer.errors import RunBundleError
from rl_fzerox.core.manager.transfer.files import archive_path_for_run_file, run_files
from rl_fzerox.core.manager.transfer.models import (
    RunBundleEvent,
    RunBundleFile,
    RunBundleImportResult,
    RunBundleLayout,
    RunBundleManifest,
    RunBundleRecord,
    RunBundleRuntime,
)
from rl_fzerox.core.manager.transfer.payload import extract_run_payload, read_manifest
from rl_fzerox.core.manager.transfer.rewrite import (
    path_replacements,
    rewrite_imported_manifest_paths,
    rewritten_optional_path,
)
from rl_fzerox.core.runtime_spec.paths import project_root_dir

MAX_RUN_BUNDLE_PAYLOAD_BYTES = 16 * 1024 * 1024 * 1024
MAX_RUN_BUNDLE_PAYLOAD_FILES = 100_000


def default_run_export_path(run_id: str) -> Path:
    """Return the default local export path for one run bundle."""

    return project_root_dir() / "local" / "exports" / f"{run_id}.zip"


def export_run_bundle(
    *,
    store: ManagerStore,
    run_id: str,
    output_path: Path | None = None,
    allow_running: bool = False,
) -> Path:
    """Write one portable run bundle and return its archive path."""

    run = store.get_run(run_id)
    if run is None:
        raise RunBundleError(f"run {run_id!r} does not exist")
    if run.status == "running" and not allow_running:
        raise RunBundleError("refusing to export a running run; stop it or pass allow_running")
    if not run.run_dir.is_dir():
        raise RunBundleError(f"run directory does not exist: {run.run_dir}")

    bundle_path = (output_path or default_run_export_path(run.id)).expanduser().resolve()
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    layout = RunBundleLayout()
    files = tuple(run_files(run.run_dir))
    manifest = RunBundleManifest(
        format_name=layout.format_name,
        schema_version=layout.schema_version,
        exported_at=store.utc_now(),
        project_root=str(project_root_dir()),
        run=_record_for_run(store, run),
        files=tuple(
            RunBundleFile(
                path=archive_path_for_run_file(layout, file_path, run_dir=run.run_dir),
                size_bytes=file_path.stat().st_size,
            )
            for file_path in files
        ),
    )

    temporary_path = bundle_path.with_name(f".{bundle_path.name}.tmp")
    if temporary_path.exists():
        temporary_path.unlink()
    with zipfile.ZipFile(temporary_path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            layout.manifest_path,
            manifest.model_dump_json(indent=2),
        )
        for file_path in files:
            archive.write(
                file_path,
                archive_path_for_run_file(layout, file_path, run_dir=run.run_dir),
            )
    temporary_path.replace(bundle_path)
    return bundle_path


def import_run_bundle(
    *,
    store: ManagerStore,
    bundle_path: Path,
    run_id: str | None = None,
    managed_runs_root: Path | None = None,
) -> RunBundleImportResult:
    """Import one portable run bundle into the manager registry."""

    archive_path = bundle_path.expanduser().resolve()
    if not archive_path.is_file():
        raise RunBundleError(f"bundle does not exist: {archive_path}")

    layout = RunBundleLayout()
    target_run_dir: Path | None = None
    extracted_payload = False
    try:
        with zipfile.ZipFile(archive_path, mode="r") as archive:
            manifest = read_manifest(archive, layout=layout)
            target_run_id = (run_id or manifest.run.id).strip()
            if not target_run_id:
                raise RunBundleError("target run id cannot be empty")
            target_lineage_id = _target_lineage_id(manifest.run, target_run_id)
            target_runs_root = store.manager_runs_root(output_root=managed_runs_root)
            target_run_dir = (
                (target_runs_root / target_lineage_id / target_run_id).expanduser().resolve()
            )
            _assert_import_target_available(store, run_id=target_run_id, run_dir=target_run_dir)
            extract_run_payload(
                archive,
                layout=layout,
                target_run_dir=target_run_dir,
                max_payload_bytes=MAX_RUN_BUNDLE_PAYLOAD_BYTES,
                max_payload_files=MAX_RUN_BUNDLE_PAYLOAD_FILES,
            )
            extracted_payload = True

        replacements = path_replacements(
            manifest=manifest,
            target_run_dir=target_run_dir,
            target_runs_root=target_runs_root,
        )
        if target_run_id != manifest.run.id:
            replacements = (*replacements, (manifest.run.id, target_run_id))
        rewrite_imported_manifest_paths(target_run_dir, replacements=replacements)
        imported_status = _imported_status(manifest.run.status)
        imported_at = store.utc_now()
        imported_config = ManagedRunConfig.model_validate(manifest.run.config)
        source_snapshot_dir = rewritten_optional_path(
            manifest.run.source_snapshot_dir,
            replacements=replacements,
        )

        store.initialize()
        with store._orm_session() as session:
            parent_run_id = _existing_run_id(session, manifest.run.parent_run_id)
            source_run_id = _existing_run_id(session, manifest.run.source_run_id)
            config_snapshot = create_config_snapshot(
                session,
                kind="import",
                config=imported_config,
                created_at=imported_at,
            )
            imported_run = ManagedRun(
                id=target_run_id,
                name=manifest.run.name,
                status=imported_status,
                config=imported_config,
                config_hash=config_snapshot.config_hash,
                run_dir=target_run_dir,
                lineage_id=target_lineage_id,
                lineage_step_offset=manifest.run.lineage_step_offset,
                parent_run_id=parent_run_id,
                source_run_id=source_run_id,
                source_artifact=manifest.run.source_artifact if source_run_id is not None else None,
                source_snapshot_dir=None
                if source_snapshot_dir is None
                else Path(source_snapshot_dir),
                source_num_timesteps=(
                    manifest.run.source_num_timesteps if source_run_id is not None else None
                ),
                created_at=manifest.run.created_at,
                started_at=manifest.run.started_at,
                stopped_at=(
                    manifest.run.stopped_at
                    if imported_status == manifest.run.status
                    else imported_at
                ),
            )
            insert_run(session, run=imported_run, config_snapshot_id=config_snapshot.id)
            session.flush()
            for group_name in manifest.run.lineage_groups:
                session.merge(
                    LineageGroupModel(
                        lineage_id=target_lineage_id,
                        group_name=group_name,
                        updated_at=imported_at,
                    )
                )
            _insert_runtime(session, run_id=target_run_id, runtime=manifest.run.runtime)
            for event in manifest.run.events:
                append_run_event(
                    session,
                    run_id=target_run_id,
                    created_at=event.created_at,
                    kind=event.kind,
                    message=event.message,
                )
            append_run_event(
                session,
                run_id=target_run_id,
                created_at=imported_at,
                kind="imported",
                message=f"run imported from {archive_path.name}",
            )
    except SqlAlchemyIntegrityError as error:
        if extracted_payload and target_run_dir is not None:
            shutil.rmtree(target_run_dir, ignore_errors=True)
        raise RunBundleError(f"could not import run bundle: {error}") from error
    except Exception:
        if extracted_payload and target_run_dir is not None:
            shutil.rmtree(target_run_dir, ignore_errors=True)
        raise

    store.rebuild_tensorboard_views()
    return RunBundleImportResult(
        run_id=target_run_id,
        run_dir=str(target_run_dir),
        imported_status=imported_status,
    )


def _record_for_run(store: ManagerStore, run: ManagedRun) -> RunBundleRecord:
    runtime = _runtime_for_bundle(run.runtime)
    return RunBundleRecord(
        id=run.id,
        name=run.name,
        status=run.status,
        config=run.config.model_dump(mode="json"),
        run_dir=str(run.run_dir),
        lineage_id=run.lineage_id,
        lineage_groups=run.lineage_groups,
        lineage_step_offset=run.lineage_step_offset,
        parent_run_id=run.parent_run_id,
        source_run_id=run.source_run_id,
        source_artifact=run.source_artifact,
        source_snapshot_dir=None
        if run.source_snapshot_dir is None
        else str(run.source_snapshot_dir),
        source_num_timesteps=run.source_num_timesteps,
        created_at=run.created_at,
        started_at=run.started_at,
        stopped_at=run.stopped_at,
        runtime=runtime,
        events=_events_for_bundle(store, run.id),
    )


def _runtime_for_bundle(runtime: ManagedRunRuntime | None) -> RunBundleRuntime | None:
    if runtime is None:
        return None
    return RunBundleRuntime(
        total_timesteps=runtime.total_timesteps,
        num_timesteps=runtime.num_timesteps,
        progress_fraction=runtime.progress_fraction,
        updated_at=runtime.updated_at,
        fps=runtime.fps,
        episode_reward_mean=runtime.episode_reward_mean,
        episode_length_mean=runtime.episode_length_mean,
        approx_kl=runtime.approx_kl,
        entropy_loss=runtime.entropy_loss,
        value_loss=runtime.value_loss,
        policy_gradient_loss=runtime.policy_gradient_loss,
    )


def _events_for_bundle(store: ManagerStore, run_id: str) -> tuple[RunBundleEvent, ...]:
    store.initialize()
    with store._connect() as connection:
        rows = connection.execute(
            """
            SELECT created_at, kind, message
            FROM run_events
            WHERE run_id = ?
            ORDER BY created_at ASC, id ASC
            """,
            (run_id,),
        ).fetchall()
    return tuple(
        RunBundleEvent(
            created_at=str(row["created_at"]),
            kind=str(row["kind"]),
            message=str(row["message"]),
        )
        for row in rows
    )


def _assert_import_target_available(store: ManagerStore, *, run_id: str, run_dir: Path) -> None:
    if store.get_run(run_id) is not None:
        raise RunBundleError(f"run {run_id!r} already exists in this manager database")
    if run_dir.exists():
        raise RunBundleError(f"target run directory already exists: {run_dir}")


def _target_lineage_id(run: RunBundleRecord, target_run_id: str) -> str:
    if run.lineage_id == run.id:
        return target_run_id
    return run.lineage_id


def _imported_status(status: RunStatus) -> RunStatus:
    match status:
        case "running" | "paused" | "created":
            return "stopped"
        case "archived":
            return "archived"
        case "stopped" | "finished" | "failed":
            return status


def _existing_run_id(session: Session, run_id: str | None) -> str | None:
    if run_id is None:
        return None
    run = session.get(RunModel, run_id)
    return None if run is None else run.id


def _insert_runtime(
    session: Session,
    *,
    run_id: str,
    runtime: RunBundleRuntime | None,
) -> None:
    if runtime is None:
        return
    session.add(
        RunRuntimeModel(
            run_id=run_id,
            total_timesteps=runtime.total_timesteps,
            num_timesteps=runtime.num_timesteps,
            progress_fraction=runtime.progress_fraction,
            updated_at=runtime.updated_at,
            fps=runtime.fps,
            episode_reward_mean=runtime.episode_reward_mean,
            episode_length_mean=runtime.episode_length_mean,
            approx_kl=runtime.approx_kl,
            entropy_loss=runtime.entropy_loss,
            value_loss=runtime.value_loss,
            policy_gradient_loss=runtime.policy_gradient_loss,
        )
    )
