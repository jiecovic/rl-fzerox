# src/rl_fzerox/apps/run_manager/api/handlers/transfer.py
from __future__ import annotations

import shutil
from pathlib import Path
from uuid import uuid4

from fastapi import HTTPException, UploadFile

from rl_fzerox.apps.run_manager.api.handlers.common import run_response
from rl_fzerox.core.manager import ManagerStore
from rl_fzerox.core.manager.transfer import RunBundleError, export_run_bundle, import_run_bundle


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
            shutil.copyfileobj(bundle.file, target, length=1024 * 1024)
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
