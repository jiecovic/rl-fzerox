# src/rl_fzerox/apps/run_manager/api/handlers/checkpoints.py
"""Handlers for official checkpoint catalog listing and installation."""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from typing import Protocol
from urllib.request import Request, urlopen
from uuid import uuid4

from fastapi import HTTPException

from rl_fzerox.apps.run_manager.api.payloads.checkpoints import (
    CheckpointCatalogPayload,
    checkpoint_catalog_payload,
    published_checkpoint_payload,
)
from rl_fzerox.core.manager import ManagedPublishedCheckpoint, ManagerStore
from rl_fzerox.core.manager.checkpoints import (
    CheckpointBundleImportError,
    CheckpointCatalog,
    CheckpointCatalogEntry,
    CheckpointCatalogError,
    default_checkpoint_catalog_path,
    parse_checkpoint_catalog_json,
)

MAX_CHECKPOINT_CATALOG_DOWNLOAD_BYTES = 4 * 1024 * 1024 * 1024
_DOWNLOAD_CHUNK_BYTES = 1024 * 1024


def checkpoint_catalog_response(store: ManagerStore) -> CheckpointCatalogPayload:
    catalog = _load_checkpoint_catalog()
    return checkpoint_catalog_payload(
        catalog,
        installed_checkpoints=store.list_published_checkpoints(),
    )


def install_catalog_checkpoint_response(
    store: ManagerStore,
    *,
    checkpoint_id: str,
    version: str,
) -> dict[str, object]:
    catalog = _load_checkpoint_catalog()
    entry = _catalog_entry(catalog, checkpoint_id=checkpoint_id, version=version)
    installed = _installed_checkpoint_for_entry(store, entry)
    if installed is not None:
        return {
            "status": "already_installed",
            "checkpoint": published_checkpoint_payload(installed),
        }

    try:
        bundle_path = _download_catalog_bundle(
            entry,
            download_dir=store.db_path.parent / "downloads" / "checkpoints",
        )
        checkpoint = store.import_published_checkpoint_bundle(bundle_path=bundle_path)
    except (CheckpointBundleImportError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "status": "installed",
        "checkpoint": published_checkpoint_payload(checkpoint),
    }


def _load_checkpoint_catalog() -> CheckpointCatalog:
    try:
        return parse_checkpoint_catalog_json(
            default_checkpoint_catalog_path().read_text(encoding="utf-8")
        )
    except (CheckpointCatalogError, OSError) as exc:
        raise HTTPException(
            status_code=500,
            detail=f"checkpoint catalog unavailable: {exc}",
        ) from exc


def _catalog_entry(
    catalog: CheckpointCatalog,
    *,
    checkpoint_id: str,
    version: str,
) -> CheckpointCatalogEntry:
    for entry in catalog.entries:
        if entry.id == checkpoint_id and entry.version == version:
            return entry
    raise HTTPException(status_code=404, detail="checkpoint catalog entry not found")


def _installed_checkpoint_for_entry(
    store: ManagerStore,
    entry: CheckpointCatalogEntry,
) -> ManagedPublishedCheckpoint | None:
    for checkpoint in store.list_published_checkpoints():
        if checkpoint.checkpoint_id == entry.id and checkpoint.version == entry.version:
            return checkpoint
    return None


def _download_catalog_bundle(
    entry: CheckpointCatalogEntry,
    *,
    download_dir: Path,
) -> Path:
    download_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = download_dir / entry.bundle.filename
    if _existing_bundle_is_valid(bundle_path, entry=entry):
        return bundle_path

    temporary_path = download_dir / f".download-{uuid4().hex}.zip"
    try:
        request = Request(
            entry.bundle.url,
            headers={"User-Agent": "rl-fzerox-run-manager"},
        )
        with urlopen(request, timeout=120) as response:
            with temporary_path.open("wb") as target:
                shutil.copyfileobj(
                    _DownloadLimitReader(response, max_bytes=entry.bundle.size_bytes),
                    target,
                    length=_DOWNLOAD_CHUNK_BYTES,
                )
        _verify_downloaded_bundle(temporary_path, entry=entry)
        temporary_path.replace(bundle_path)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"checkpoint download failed: {exc}") from exc
    finally:
        temporary_path.unlink(missing_ok=True)

    return bundle_path


def _existing_bundle_is_valid(path: Path, *, entry: CheckpointCatalogEntry) -> bool:
    if not path.is_file():
        return False
    try:
        _verify_downloaded_bundle(path, entry=entry)
    except HTTPException:
        return False
    return True


def _verify_downloaded_bundle(path: Path, *, entry: CheckpointCatalogEntry) -> None:
    actual_size = path.stat().st_size
    if actual_size != entry.bundle.size_bytes:
        raise HTTPException(
            status_code=502,
            detail=(
                "checkpoint download size mismatch: "
                f"expected {entry.bundle.size_bytes}, got {actual_size}"
            ),
        )
    actual_sha256 = _sha256_file(path)
    if actual_sha256 != entry.bundle.sha256:
        raise HTTPException(
            status_code=502,
            detail="checkpoint download sha256 mismatch",
        )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(_DOWNLOAD_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


class _Readable(Protocol):
    def read(self, size: int = -1) -> bytes: ...


class _DownloadLimitReader:
    """File-like adapter that stops oversized downloads before disk growth."""

    def __init__(self, source: _Readable, *, max_bytes: int) -> None:
        self._source = source
        self._remaining = max_bytes

    def read(self, size: int = -1) -> bytes:
        if self._remaining < 0:
            raise HTTPException(status_code=502, detail="checkpoint download is too large")
        read_size = size if size >= 0 else _DOWNLOAD_CHUNK_BYTES
        chunk = self._source.read(read_size)
        self._remaining -= len(chunk)
        if self._remaining < 0:
            raise HTTPException(status_code=502, detail="checkpoint download is too large")
        return chunk
