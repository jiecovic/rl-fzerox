# src/rl_fzerox/core/manager/transfer/archive.py
from __future__ import annotations

import shutil
import sqlite3
import stat
import zipfile
from collections.abc import Iterable
from pathlib import Path, PurePosixPath

from rl_fzerox.core.manager.models import ManagedRun, ManagedRunRuntime, RunStatus
from rl_fzerox.core.manager.run_spec import ManagedRunConfig
from rl_fzerox.core.manager.storage.serialization import config_hash, config_json
from rl_fzerox.core.manager.store import ManagerStore
from rl_fzerox.core.manager.transfer.models import (
    RunBundleEvent,
    RunBundleFile,
    RunBundleImportResult,
    RunBundleLayout,
    RunBundleManifest,
    RunBundleRecord,
    RunBundleRuntime,
)
from rl_fzerox.core.runtime_spec.paths import project_root_dir


class RunBundleError(RuntimeError):
    """Raised when a run bundle cannot be exported or imported safely."""


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
    files = tuple(_run_files(run.run_dir))
    manifest = RunBundleManifest(
        format_name=layout.format_name,
        schema_version=layout.schema_version,
        exported_at=store.utc_now(),
        project_root=str(project_root_dir()),
        run=_record_for_run(store, run),
        files=tuple(
            RunBundleFile(
                path=_archive_path_for_run_file(layout, file_path, run_dir=run.run_dir),
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
                _archive_path_for_run_file(layout, file_path, run_dir=run.run_dir),
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
            manifest = _read_manifest(archive, layout=layout)
            target_run_id = (run_id or manifest.run.id).strip()
            if not target_run_id:
                raise RunBundleError("target run id cannot be empty")
            target_lineage_id = _target_lineage_id(manifest.run, target_run_id)
            target_runs_root = store.manager_runs_root(output_root=managed_runs_root)
            target_run_dir = (
                (target_runs_root / target_lineage_id / target_run_id).expanduser().resolve()
            )
            _assert_import_target_available(store, run_id=target_run_id, run_dir=target_run_dir)
            _extract_run_payload(archive, layout=layout, target_run_dir=target_run_dir)
            extracted_payload = True

        replacements = _path_replacements(
            manifest=manifest,
            target_run_dir=target_run_dir,
            target_runs_root=target_runs_root,
        )
        if target_run_id != manifest.run.id:
            replacements = (*replacements, (manifest.run.id, target_run_id))
        _rewrite_imported_manifest_paths(target_run_dir, replacements=replacements)
        imported_status = _imported_status(manifest.run.status)
        imported_at = store.utc_now()
        imported_config = ManagedRunConfig.model_validate(manifest.run.config)
        source_snapshot_dir = _rewritten_optional_path(
            manifest.run.source_snapshot_dir,
            replacements=replacements,
        )

        store.initialize()
        with store._connect() as connection:
            parent_run_id = _existing_run_id(connection, manifest.run.parent_run_id)
            source_run_id = _existing_run_id(connection, manifest.run.source_run_id)
            connection.execute(
                """
                INSERT INTO runs(
                    id,
                    name,
                    status,
                    config_json,
                    config_hash,
                    run_dir,
                    lineage_id,
                    lineage_step_offset,
                    parent_run_id,
                    source_run_id,
                    source_artifact,
                    source_snapshot_dir,
                    source_num_timesteps,
                    created_at,
                    started_at,
                    stopped_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    target_run_id,
                    manifest.run.name,
                    imported_status,
                    config_json(imported_config),
                    config_hash(imported_config),
                    str(target_run_dir),
                    target_lineage_id,
                    manifest.run.lineage_step_offset,
                    parent_run_id,
                    source_run_id,
                    manifest.run.source_artifact if source_run_id is not None else None,
                    source_snapshot_dir,
                    manifest.run.source_num_timesteps if source_run_id is not None else None,
                    manifest.run.created_at,
                    manifest.run.started_at,
                    manifest.run.stopped_at
                    if imported_status == manifest.run.status
                    else imported_at,
                ),
            )
            for group_name in manifest.run.lineage_groups:
                connection.execute(
                    """
                    INSERT OR IGNORE INTO lineage_groups(lineage_id, group_name, updated_at)
                    VALUES (?, ?, ?)
                    """,
                    (target_lineage_id, group_name, imported_at),
                )
            _insert_runtime(connection, run_id=target_run_id, runtime=manifest.run.runtime)
            for event in manifest.run.events:
                connection.execute(
                    """
                    INSERT INTO run_events(run_id, created_at, kind, message)
                    VALUES (?, ?, ?, ?)
                    """,
                    (target_run_id, event.created_at, event.kind, event.message),
                )
            connection.execute(
                """
                INSERT INTO run_events(run_id, created_at, kind, message)
                VALUES (?, ?, ?, ?)
                """,
                (
                    target_run_id,
                    imported_at,
                    "imported",
                    f"run imported from {archive_path.name}",
                ),
            )
    except sqlite3.IntegrityError as error:
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


def _run_files(run_dir: Path) -> Iterable[Path]:
    for file_path in sorted(run_dir.rglob("*")):
        if file_path.is_symlink() or not file_path.is_file():
            continue
        yield file_path


def _archive_path_for_run_file(layout: RunBundleLayout, file_path: Path, *, run_dir: Path) -> str:
    return str(PurePosixPath(layout.payload_dir, file_path.relative_to(run_dir).as_posix()))


def _read_manifest(
    archive: zipfile.ZipFile,
    *,
    layout: RunBundleLayout,
) -> RunBundleManifest:
    try:
        raw_manifest = archive.read(layout.manifest_path)
    except KeyError as error:
        raise RunBundleError("bundle manifest is missing") from error
    manifest = RunBundleManifest.model_validate_json(raw_manifest)
    if manifest.format_name != layout.format_name:
        raise RunBundleError(f"unsupported bundle format: {manifest.format_name!r}")
    if manifest.schema_version != layout.schema_version:
        raise RunBundleError(f"unsupported bundle schema version: {manifest.schema_version}")
    return manifest


def _assert_import_target_available(store: ManagerStore, *, run_id: str, run_dir: Path) -> None:
    if store.get_run(run_id) is not None:
        raise RunBundleError(f"run {run_id!r} already exists in this manager database")
    if run_dir.exists():
        raise RunBundleError(f"target run directory already exists: {run_dir}")


def _extract_run_payload(
    archive: zipfile.ZipFile,
    *,
    layout: RunBundleLayout,
    target_run_dir: Path,
) -> None:
    target_run_dir.mkdir(parents=True, exist_ok=False)
    try:
        for info in archive.infolist():
            if info.filename == layout.manifest_path:
                continue
            relative_path = _safe_payload_relative_path(info, layout=layout)
            if relative_path is None:
                continue
            target_path = target_run_dir.joinpath(*relative_path.parts)
            if not target_path.resolve().is_relative_to(target_run_dir.resolve()):
                raise RunBundleError(f"unsafe archive member: {info.filename}")
            if info.is_dir():
                target_path.mkdir(parents=True, exist_ok=True)
                continue
            target_path.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info, mode="r") as source, target_path.open("wb") as target:
                shutil.copyfileobj(source, target)
    except Exception:
        shutil.rmtree(target_run_dir, ignore_errors=True)
        raise


def _safe_payload_relative_path(
    info: zipfile.ZipInfo,
    *,
    layout: RunBundleLayout,
) -> PurePosixPath | None:
    mode = info.external_attr >> 16
    if stat.S_ISLNK(mode):
        raise RunBundleError(f"bundle contains unsupported symlink: {info.filename}")
    member_path = PurePosixPath(info.filename)
    if member_path.is_absolute() or ".." in member_path.parts:
        raise RunBundleError(f"unsafe archive member: {info.filename}")
    if not member_path.parts or member_path.parts[0] != layout.payload_dir:
        raise RunBundleError(f"unexpected archive member outside payload: {info.filename}")
    relative_parts = member_path.parts[1:]
    if not relative_parts:
        return None
    return PurePosixPath(*relative_parts)


def _path_replacements(
    *,
    manifest: RunBundleManifest,
    target_run_dir: Path,
    target_runs_root: Path,
) -> tuple[tuple[str, str], ...]:
    source_run_dir = Path(manifest.run.run_dir)
    source_lineage_dir = source_run_dir.parent
    source_runs_root = source_lineage_dir.parent
    replacements = {
        str(source_run_dir): str(target_run_dir),
        str(source_lineage_dir): str(target_run_dir.parent),
        str(source_runs_root): str(target_runs_root),
        manifest.project_root: str(project_root_dir()),
    }
    return tuple(
        sorted(
            (
                (source, target)
                for source, target in replacements.items()
                if source and source != target
            ),
            key=lambda item: len(item[0]),
            reverse=True,
        )
    )


def _rewrite_imported_manifest_paths(
    run_dir: Path,
    *,
    replacements: tuple[tuple[str, str], ...],
) -> None:
    for file_path in _rewrite_candidate_files(run_dir):
        try:
            text = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        rewritten = _rewrite_path_text(text, replacements=replacements)
        if rewritten != text:
            file_path.write_text(rewritten, encoding="utf-8")


def _rewrite_candidate_files(run_dir: Path) -> Iterable[Path]:
    suffixes = {".json", ".yaml", ".yml", ".txt", ".toml"}
    for file_path in _run_files(run_dir):
        if file_path.suffix.lower() in suffixes:
            yield file_path


def _rewrite_path_text(text: str, *, replacements: tuple[tuple[str, str], ...]) -> str:
    rewritten = text
    for source, target in replacements:
        rewritten = rewritten.replace(source, target)
    return rewritten


def _rewritten_optional_path(
    value: str | None,
    *,
    replacements: tuple[tuple[str, str], ...],
) -> str | None:
    if value is None:
        return None
    rewritten = _rewrite_path_text(value, replacements=replacements)
    return rewritten if Path(rewritten).exists() else None


def _target_lineage_id(run: RunBundleRecord, target_run_id: str) -> str:
    if run.lineage_id == run.id:
        return target_run_id
    return run.lineage_id


def _imported_status(status: RunStatus) -> RunStatus:
    match status:
        case "running" | "paused" | "created":
            return "stopped"
        case "stopped" | "finished" | "failed":
            return status


def _existing_run_id(connection: sqlite3.Connection, run_id: str | None) -> str | None:
    if run_id is None:
        return None
    row = connection.execute("SELECT id FROM runs WHERE id = ?", (run_id,)).fetchone()
    return None if row is None else str(row["id"])


def _insert_runtime(
    connection: sqlite3.Connection,
    *,
    run_id: str,
    runtime: RunBundleRuntime | None,
) -> None:
    if runtime is None:
        return
    connection.execute(
        """
        INSERT INTO run_runtime(
            run_id,
            total_timesteps,
            num_timesteps,
            progress_fraction,
            updated_at,
            fps,
            episode_reward_mean,
            episode_length_mean,
            approx_kl,
            entropy_loss,
            value_loss,
            policy_gradient_loss
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            run_id,
            runtime.total_timesteps,
            runtime.num_timesteps,
            runtime.progress_fraction,
            runtime.updated_at,
            runtime.fps,
            runtime.episode_reward_mean,
            runtime.episode_length_mean,
            runtime.approx_kl,
            runtime.entropy_loss,
            runtime.value_loss,
            runtime.policy_gradient_loss,
        ),
    )
