# src/rl_fzerox/apps/run_manager/api/handlers/lineages.py
from __future__ import annotations

from fastapi import HTTPException

from rl_fzerox.apps.run_manager.api.contracts import UpdateLineageGroupsRequest
from rl_fzerox.apps.run_manager.api.payloads.metrics import tensorboard_view_group_payload
from rl_fzerox.core.manager import ManagerStore


def delete_lineage_payload(store: ManagerStore, lineage_id: str) -> dict[str, bool]:
    try:
        deleted = store.delete_lineage(lineage_id)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    if not deleted:
        raise HTTPException(status_code=404, detail="lineage not found")
    store.rebuild_tensorboard_views()
    return {"deleted": True}


def update_lineage_groups_payload(
    store: ManagerStore,
    lineage_id: str,
    request: UpdateLineageGroupsRequest,
) -> dict[str, object]:
    try:
        group_names = store.update_lineage_groups(
            lineage_id=lineage_id,
            group_names=request.group_names,
        )
    except ValueError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    view_groups = store.rebuild_tensorboard_views()
    return {
        "lineage_id": lineage_id,
        "lineage_groups": list(group_names),
        "tensorboard_views": [tensorboard_view_group_payload(group) for group in view_groups],
    }


def rebuild_tensorboard_views_payload(store: ManagerStore) -> dict[str, object]:
    view_groups = store.rebuild_tensorboard_views()
    return {
        "tensorboard_views": [tensorboard_view_group_payload(group) for group in view_groups],
    }
