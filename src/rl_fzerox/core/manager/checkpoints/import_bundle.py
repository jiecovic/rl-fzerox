# src/rl_fzerox/core/manager/checkpoints/import_bundle.py
"""Validate and install downloaded checkpoint release bundles.

The importer is the first local trust boundary for public checkpoint ZIPs. It
does not create manager records yet; it only verifies the manifest, hashes, ZIP
member layout, and payload sizes before copying curated files into manager-owned
local storage. SQLite ownership is added by the next checkpoint registry step.
"""

from __future__ import annotations

import hashlib
import shutil
import stat
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from rl_fzerox.core.manager.checkpoints.manifest import (
    CHECKPOINT_BUNDLE_LAYOUT,
    CheckpointBundleFile,
    CheckpointBundleManifest,
    CheckpointBundleManifestError,
    parse_checkpoint_bundle_manifest_json,
    serialize_checkpoint_bundle_manifest_json,
)
from rl_fzerox.core.manager.registry.common import slugify

MAX_CHECKPOINT_BUNDLE_PAYLOAD_BYTES = 4 * 1024 * 1024 * 1024
MAX_CHECKPOINT_BUNDLE_PAYLOAD_FILES = 16
_DEFAULT_MANAGER_DB_PATH = Path("local/manager/runs.db").resolve()


@dataclass(frozen=True)
class CheckpointBundleImportResult:
    """Result of installing one validated checkpoint bundle payload."""

    checkpoint_id: str
    version: str
    import_dir: Path
    manifest: CheckpointBundleManifest
    files: tuple[Path, ...]


class CheckpointBundleImportError(ValueError):
    """Raised when a checkpoint bundle cannot be imported safely."""


def default_imported_checkpoint_root(*, db_path: Path | None = None) -> Path:
    """Return the default local storage root for imported checkpoint payloads."""

    manager_db_path = (db_path or _DEFAULT_MANAGER_DB_PATH).expanduser().resolve()
    return manager_db_path.parent / "checkpoints"


def read_checkpoint_bundle_manifest(bundle_path: Path) -> CheckpointBundleManifest:
    """Read one bundle manifest without extracting payload files."""

    archive_path = bundle_path.expanduser().resolve()
    if not archive_path.is_file():
        raise CheckpointBundleImportError(f"checkpoint bundle does not exist: {archive_path}")
    try:
        with zipfile.ZipFile(archive_path, mode="r") as archive:
            return _read_manifest(archive)
    except CheckpointBundleImportError:
        raise
    except Exception as exc:
        raise CheckpointBundleImportError(f"could not read checkpoint bundle: {exc}") from exc


def validate_checkpoint_bundle_archive(
    *,
    bundle_path: Path,
    max_payload_bytes: int = MAX_CHECKPOINT_BUNDLE_PAYLOAD_BYTES,
    max_payload_files: int = MAX_CHECKPOINT_BUNDLE_PAYLOAD_FILES,
) -> CheckpointBundleManifest:
    """Validate one checkpoint ZIP without copying files into manager storage."""

    archive_path = bundle_path.expanduser().resolve()
    if not archive_path.is_file():
        raise CheckpointBundleImportError(f"checkpoint bundle does not exist: {archive_path}")
    try:
        with zipfile.ZipFile(archive_path, mode="r") as archive:
            manifest = _read_manifest(archive)
            _validate_archive_members(
                archive,
                manifest=manifest,
                max_payload_bytes=max_payload_bytes,
                max_payload_files=max_payload_files,
            )
            _verify_payload_hashes(archive, manifest=manifest)
            return manifest
    except CheckpointBundleImportError:
        raise
    except Exception as exc:
        raise CheckpointBundleImportError(f"could not validate checkpoint bundle: {exc}") from exc


def import_checkpoint_bundle(
    *,
    bundle_path: Path,
    target_root: Path | None = None,
    overwrite: bool = False,
    max_payload_bytes: int = MAX_CHECKPOINT_BUNDLE_PAYLOAD_BYTES,
    max_payload_files: int = MAX_CHECKPOINT_BUNDLE_PAYLOAD_FILES,
) -> CheckpointBundleImportResult:
    """Validate and copy one checkpoint bundle into local checkpoint storage."""

    archive_path = bundle_path.expanduser().resolve()
    if not archive_path.is_file():
        raise CheckpointBundleImportError(f"checkpoint bundle does not exist: {archive_path}")

    root = (target_root or default_imported_checkpoint_root()).expanduser().resolve()
    temporary_dir: Path | None = None
    try:
        with zipfile.ZipFile(archive_path, mode="r") as archive:
            manifest = _read_manifest(archive)
            _validate_archive_members(
                archive,
                manifest=manifest,
                max_payload_bytes=max_payload_bytes,
                max_payload_files=max_payload_files,
            )
            import_dir = _import_dir(root, manifest)
            _assert_target_available(import_dir, overwrite=overwrite)
            temporary_dir = import_dir.with_name(f".{import_dir.name}.tmp")
            _reset_temporary_dir(temporary_dir)
            files = _extract_payloads(archive, manifest=manifest, target_dir=temporary_dir)
            relative_files = tuple(file.relative_to(temporary_dir) for file in files)
            _write_manifest(temporary_dir, manifest)
        _replace_import_dir(temporary_dir, import_dir, overwrite=overwrite)
    except CheckpointBundleImportError:
        if temporary_dir is not None:
            shutil.rmtree(temporary_dir, ignore_errors=True)
        raise
    except Exception as exc:
        if temporary_dir is not None:
            shutil.rmtree(temporary_dir, ignore_errors=True)
        raise CheckpointBundleImportError(f"could not import checkpoint bundle: {exc}") from exc

    return CheckpointBundleImportResult(
        checkpoint_id=manifest.checkpoint.id,
        version=manifest.checkpoint.version,
        import_dir=import_dir,
        manifest=manifest,
        files=tuple(import_dir / file for file in relative_files),
    )


def _read_manifest(archive: zipfile.ZipFile) -> CheckpointBundleManifest:
    try:
        raw_manifest = archive.read(CHECKPOINT_BUNDLE_LAYOUT.manifest_path)
    except KeyError as exc:
        raise CheckpointBundleImportError("checkpoint bundle manifest is missing") from exc
    try:
        return parse_checkpoint_bundle_manifest_json(raw_manifest.decode("utf-8"))
    except (CheckpointBundleManifestError, UnicodeDecodeError) as exc:
        raise CheckpointBundleImportError(f"invalid checkpoint bundle manifest: {exc}") from exc


def _validate_archive_members(
    archive: zipfile.ZipFile,
    *,
    manifest: CheckpointBundleManifest,
    max_payload_bytes: int,
    max_payload_files: int,
) -> None:
    expected_files = frozenset(
        {CHECKPOINT_BUNDLE_LAYOUT.manifest_path, *(file.path for file in manifest.files)}
    )
    seen: set[str] = set()
    payload_bytes = 0
    payload_files = 0
    expected_parent_dirs = _expected_parent_dirs(expected_files)

    for info in archive.infolist():
        _validate_archive_path(info)
        if info.filename in seen:
            raise CheckpointBundleImportError(f"duplicate archive member: {info.filename}")
        seen.add(info.filename)
        if info.is_dir():
            if info.filename.rstrip("/") not in expected_parent_dirs:
                raise CheckpointBundleImportError(f"unexpected archive directory: {info.filename}")
            continue
        if info.filename not in expected_files:
            raise CheckpointBundleImportError(f"unexpected archive member: {info.filename}")
        if info.filename == CHECKPOINT_BUNDLE_LAYOUT.manifest_path:
            continue
        payload_files += 1
        if payload_files > max_payload_files:
            raise CheckpointBundleImportError(
                f"checkpoint bundle payload has more than {max_payload_files} files"
            )
        payload_bytes += info.file_size
        if payload_bytes > max_payload_bytes:
            raise CheckpointBundleImportError(
                f"checkpoint bundle payload exceeds {max_payload_bytes} bytes"
            )

    missing_files = sorted(expected_files.difference(seen))
    if missing_files:
        raise CheckpointBundleImportError(
            f"checkpoint bundle is missing payload files: {missing_files}"
        )


def _validate_archive_path(info: zipfile.ZipInfo) -> None:
    mode = info.external_attr >> 16
    if stat.S_ISLNK(mode):
        raise CheckpointBundleImportError(f"bundle contains unsupported symlink: {info.filename}")
    if "\\" in info.filename:
        raise CheckpointBundleImportError(f"unsafe archive member: {info.filename}")
    member_path = PurePosixPath(info.filename)
    if not member_path.parts or member_path.is_absolute() or ".." in member_path.parts:
        raise CheckpointBundleImportError(f"unsafe archive member: {info.filename}")


def _expected_parent_dirs(expected_files: frozenset[str]) -> frozenset[str]:
    parents: set[str] = set()
    for file_path in expected_files:
        parent = PurePosixPath(file_path).parent
        while str(parent) not in {"", "."}:
            parents.add(str(parent))
            parent = parent.parent
    return frozenset(parents)


def _import_dir(root: Path, manifest: CheckpointBundleManifest) -> Path:
    checkpoint_slug = slugify(manifest.checkpoint.id)
    version_slug = slugify(manifest.checkpoint.version)
    if not checkpoint_slug or not version_slug:
        raise CheckpointBundleImportError("checkpoint id and version must produce safe paths")
    return (root / checkpoint_slug / version_slug).resolve()


def _assert_target_available(import_dir: Path, *, overwrite: bool) -> None:
    if import_dir.exists() and not overwrite:
        raise CheckpointBundleImportError(f"import target already exists: {import_dir}")


def _reset_temporary_dir(temporary_dir: Path) -> None:
    if temporary_dir.exists():
        shutil.rmtree(temporary_dir)
    temporary_dir.mkdir(parents=True, exist_ok=False)


def _extract_payloads(
    archive: zipfile.ZipFile,
    *,
    manifest: CheckpointBundleManifest,
    target_dir: Path,
) -> tuple[Path, ...]:
    extracted: list[Path] = []
    for bundle_file in manifest.files:
        data = _verified_payload_bytes(archive, bundle_file)
        target_path = target_dir.joinpath(*PurePosixPath(bundle_file.path).parts)
        if not target_path.resolve().is_relative_to(target_dir.resolve()):
            raise CheckpointBundleImportError(f"unsafe bundle payload path: {bundle_file.path}")
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_bytes(data)
        extracted.append(target_path)
    return tuple(extracted)


def _verify_payload_hashes(
    archive: zipfile.ZipFile,
    *,
    manifest: CheckpointBundleManifest,
) -> None:
    for bundle_file in manifest.files:
        _verified_payload_bytes(archive, bundle_file)


def _verified_payload_bytes(
    archive: zipfile.ZipFile,
    bundle_file: CheckpointBundleFile,
) -> bytes:
    info = archive.getinfo(bundle_file.path)
    if info.file_size != bundle_file.size_bytes:
        raise CheckpointBundleImportError(
            f"size mismatch for {bundle_file.path}: "
            f"manifest {bundle_file.size_bytes}, archive {info.file_size}"
        )
    data = archive.read(info)
    if len(data) != bundle_file.size_bytes:
        raise CheckpointBundleImportError(
            f"size mismatch for {bundle_file.path}: "
            f"manifest {bundle_file.size_bytes}, read {len(data)}"
        )
    sha256 = hashlib.sha256(data).hexdigest()
    if sha256 != bundle_file.sha256:
        raise CheckpointBundleImportError(
            f"sha256 mismatch for {bundle_file.path}: manifest {bundle_file.sha256}"
        )
    return data


def _write_manifest(target_dir: Path, manifest: CheckpointBundleManifest) -> Path:
    manifest_path = target_dir / CHECKPOINT_BUNDLE_LAYOUT.manifest_path
    manifest_path.write_text(serialize_checkpoint_bundle_manifest_json(manifest), encoding="utf-8")
    return manifest_path


def _replace_import_dir(temporary_dir: Path, import_dir: Path, *, overwrite: bool) -> None:
    if import_dir.exists():
        if not overwrite:
            raise CheckpointBundleImportError(f"import target already exists: {import_dir}")
        shutil.rmtree(import_dir)
    import_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir.replace(import_dir)
