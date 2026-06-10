# src/rl_fzerox/apps/run_manager/api/routers/transfer.py
from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, File, Path, UploadFile
from fastapi.responses import FileResponse

from rl_fzerox.apps.run_manager.api import handlers
from rl_fzerox.apps.run_manager.api.execution import run_sync
from rl_fzerox.core.manager import ManagerStore


def create_transfer_router(store: ManagerStore) -> APIRouter:
    router = APIRouter()

    @router.get("/api/runs/{run_id}/export")
    async def export_run(
        run_id: Annotated[str, Path(min_length=1)],
    ) -> FileResponse:
        bundle_path = await run_sync(handlers.export_run_bundle_path, store, run_id)
        return FileResponse(
            bundle_path,
            filename=f"{run_id}.zip",
            media_type="application/zip",
        )

    @router.post("/api/run-imports", status_code=201)
    async def import_run(
        bundle: Annotated[UploadFile, File(description="Run export zip bundle")],
    ) -> dict[str, dict[str, object]]:
        return await run_sync(handlers.import_run_bundle_payload, store, bundle)

    return router
