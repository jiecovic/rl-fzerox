# src/rl_fzerox/core/manager/transfer/payload.py
"""Bundle manifest and payload extraction."""

from __future__ import annotations

import shutil
import stat
import zipfile
from pathlib import Path, PurePosixPath

from rl_fzerox.core.manager.transfer.errors import RunBundleError
from rl_fzerox.core.manager.transfer.models import RunBundleLayout, RunBundleManifest


def read_manifest(
    archive: zipfile.ZipFile,
    *,
    layout: RunBundleLayout,
) -> RunBundleManifest:
    """Read and validate the bundle manifest."""

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


def extract_run_payload(
    archive: zipfile.ZipFile,
    *,
    layout: RunBundleLayout,
    target_run_dir: Path,
    max_payload_bytes: int,
    max_payload_files: int,
) -> None:
    """Extract a bundle payload while enforcing path and size limits."""

    target_run_dir.mkdir(parents=True, exist_ok=False)
    try:
        payload_bytes = 0
        payload_files = 0
        for info in archive.infolist():
            if info.filename == layout.manifest_path:
                continue
            relative_path = safe_payload_relative_path(info, layout=layout)
            if relative_path is None:
                continue
            target_path = target_run_dir.joinpath(*relative_path.parts)
            if not target_path.resolve().is_relative_to(target_run_dir.resolve()):
                raise RunBundleError(f"unsafe archive member: {info.filename}")
            if info.is_dir():
                target_path.mkdir(parents=True, exist_ok=True)
                continue
            payload_files += 1
            if payload_files > max_payload_files:
                raise RunBundleError(f"bundle payload has more than {max_payload_files} files")
            payload_bytes += info.file_size
            if payload_bytes > max_payload_bytes:
                raise RunBundleError(f"bundle payload exceeds {max_payload_bytes} bytes")
            target_path.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info, mode="r") as source, target_path.open("wb") as target:
                shutil.copyfileobj(source, target)
    except Exception:
        shutil.rmtree(target_run_dir, ignore_errors=True)
        raise


def safe_payload_relative_path(
    info: zipfile.ZipInfo,
    *,
    layout: RunBundleLayout,
) -> PurePosixPath | None:
    """Return the payload-relative member path or reject unsafe archive members."""

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
