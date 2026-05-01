# src/rl_fzerox/apps/run_manager/api.py
from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated

from fastapi import FastAPI, HTTPException, Path
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict
from starlette.requests import Request

from rl_fzerox.core.manager import (
    ManagedRun,
    ManagedRunConfig,
    ManagedRunDraft,
    ManagedRunTemplate,
    ManagerStore,
)
from rl_fzerox.core.manager.architecture import (
    policy_architecture_preview,
    run_manager_config_metadata,
)


class CreateDraftRequest(BaseModel):
    """Request body for creating a SQLite-backed draft."""

    model_config = ConfigDict(extra="forbid")

    name: str
    config: ManagedRunConfig


def create_manager_api_app(store: ManagerStore) -> FastAPI:
    """Create the local REST API app for the run manager."""

    app = FastAPI(title="F-Zero X Run Manager", version="0.1.0")

    @app.exception_handler(HTTPException)
    async def handle_http_exception(_request: Request, exc: HTTPException) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})

    @app.exception_handler(RequestValidationError)
    async def handle_validation_exception(
        _request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        return JSONResponse(status_code=400, content={"error": jsonable_encoder(exc.errors())})

    @app.get("/api/health")
    def health() -> dict[str, bool]:
        return {"ok": True}

    @app.get("/api/templates")
    def templates() -> dict[str, list[dict[str, object]]]:
        return {"templates": [_template_payload(item) for item in store.list_templates()]}

    @app.get("/api/drafts")
    def drafts() -> dict[str, list[dict[str, object]]]:
        return {"drafts": [_draft_payload(item) for item in store.list_drafts()]}

    @app.get("/api/runs")
    def runs() -> dict[str, list[dict[str, object]]]:
        return {"runs": [_run_payload(item) for item in store.list_visible_runs()]}

    @app.get("/api/schema")
    def schema() -> dict[str, Mapping[str, object]]:
        return {"config": ManagedRunConfig.model_json_schema()}

    @app.get("/api/config-metadata")
    def config_metadata() -> dict[str, object]:
        return run_manager_config_metadata().model_dump(mode="json")

    @app.post("/api/policy-preview")
    def policy_preview(config: ManagedRunConfig) -> dict[str, object]:
        return policy_architecture_preview(config).model_dump(mode="json")

    @app.post("/api/drafts", status_code=201)
    def create_draft(request: CreateDraftRequest) -> dict[str, dict[str, object]]:
        name = request.name.strip()
        if not name:
            raise HTTPException(status_code=400, detail="draft name is required")
        draft = store.create_draft(name=name, config=request.config)
        return {"draft": _draft_payload(draft)}

    @app.delete("/api/drafts/{draft_id}")
    def delete_draft(
        draft_id: Annotated[str, Path(min_length=1)],
    ) -> dict[str, bool]:
        return {"deleted": store.delete_draft(draft_id)}

    return app


def _template_payload(template: ManagedRunTemplate) -> dict[str, object]:
    return {
        "id": template.id,
        "name": template.name,
        "created_at": template.created_at,
        "updated_at": template.updated_at,
        "config": template.config.model_dump(mode="json"),
    }


def _draft_payload(draft: ManagedRunDraft) -> dict[str, object]:
    return {
        "id": draft.id,
        "name": draft.name,
        "created_at": draft.created_at,
        "updated_at": draft.updated_at,
        "config": draft.config.model_dump(mode="json"),
    }


def _run_payload(run: ManagedRun) -> dict[str, object]:
    return {
        "id": run.id,
        "name": run.name,
        "status": run.status,
        "created_at": run.created_at,
        "started_at": run.started_at,
        "stopped_at": run.stopped_at,
        "parent_run_id": run.parent_run_id,
        "source_run_id": run.source_run_id,
        "source_artifact": run.source_artifact,
        "config": run.config.model_dump(mode="json"),
    }
