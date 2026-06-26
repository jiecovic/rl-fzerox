# src/rl_fzerox/apps/run_manager/api/handlers/drafts.py
from __future__ import annotations

from fastapi import HTTPException

from rl_fzerox.apps.run_manager.api.contracts import CreateDraftRequest, UpdateDraftRequest
from rl_fzerox.apps.run_manager.api.payloads.drafts import draft_payload, template_payload
from rl_fzerox.core.manager import ManagerStore
from rl_fzerox.core.manager.errors import ManagerNameConflictError


def templates_payload(store: ManagerStore) -> dict[str, list[dict[str, object]]]:
    items = store.list_templates()
    return {"templates": [template_payload(item) for item in items]}


def drafts_payload(store: ManagerStore) -> dict[str, list[dict[str, object]]]:
    items = store.list_drafts()
    return {"drafts": [draft_payload(item) for item in items]}


def create_draft_payload(
    store: ManagerStore,
    request: CreateDraftRequest,
    name: str,
) -> dict[str, dict[str, object]]:
    try:
        draft = store.create_draft(
            name=name,
            config=request.config,
            source_policy_kind=request.source_policy_kind,
            source_policy_id=request.source_policy_id,
            source_run_id=request.source_run_id,
            source_artifact=request.source_artifact,
        )
    except ManagerNameConflictError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return {"draft": draft_payload(draft)}


def update_draft_payload(
    store: ManagerStore,
    draft_id: str,
    request: UpdateDraftRequest,
    name: str,
) -> dict[str, dict[str, object]]:
    try:
        draft = store.update_draft(
            draft_id=draft_id,
            name=name,
            config=request.config,
            source_policy_kind=request.source_policy_kind,
            source_policy_id=request.source_policy_id,
            source_run_id=request.source_run_id,
            source_artifact=request.source_artifact,
        )
    except ManagerNameConflictError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    if draft is None:
        raise HTTPException(status_code=404, detail="draft not found")
    return {"draft": draft_payload(draft)}


def delete_draft_payload(store: ManagerStore, draft_id: str) -> dict[str, bool]:
    deleted = store.delete_draft(draft_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="draft not found")
    return {"deleted": True}
