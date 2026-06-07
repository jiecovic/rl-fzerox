# src/rl_fzerox/apps/run_manager/api/handlers/transfer.py
from __future__ import annotations

from pathlib import Path
from typing import BinaryIO
from uuid import uuid4

from fastapi import HTTPException, UploadFile

from rl_fzerox.apps.run_manager.api.handlers.common import run_response
from rl_fzerox.core.manager import ManagerStore
from rl_fzerox.core.manager.transfer import RunBundleError, export_run_bundle, import_run_bundle

MAX_RUN_BUNDLE_UPLOAD_BYTES = 4 * 1024 * 1024 * 1024
_UPLOAD_COPY_CHUNK_BYTES = 1024 * 1024


def export_run_bundle_path(store: ManagerStore, run_id: str) -> Path:
    try:
        return export_run_bundle(
            store=store,
            run_id=run_id,
            output_path=store.db_path.parent / "exports" / f"{run_id}.zip",
        )
    except RunBundleError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


def import_run_bundle_payload(
    store: ManagerStore,
    bundle: UploadFile,
) -> dict[str, dict[str, object]]:
    import_dir = store.db_path.parent / "imports"
    import_dir.mkdir(parents=True, exist_ok=True)
    temporary_path = import_dir / f".upload-{uuid4().hex}.zip"
    try:
        with temporary_path.open("wb") as target:
            bundle.file.seek(0)
            _copy_upload_with_limit(
                bundle.file,
                target,
                max_bytes=MAX_RUN_BUNDLE_UPLOAD_BYTES,
            )
        result = import_run_bundle(store=store, bundle_path=temporary_path)
    except RunBundleError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    finally:
        temporary_path.unlink(missing_ok=True)
        bundle.file.close()

    run = store.get_run(result.run_id)
    if run is None:
        raise HTTPException(status_code=500, detail="imported run is missing")
    return run_response(store, run)


def _copy_upload_with_limit(
    source: BinaryIO,
    target: BinaryIO,
    *,
    max_bytes: int,
) -> None:
    copied_bytes = 0
    while True:
        chunk = source.read(_UPLOAD_COPY_CHUNK_BYTES)
        if not chunk:
            return
        copied_bytes += len(chunk)
        if copied_bytes > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"run bundle upload exceeds {max_bytes} bytes",
            )
        target.write(chunk)
