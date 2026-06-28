# src/rl_fzerox/apps/run_manager/api/handlers/common.py
from __future__ import annotations

from typing import Literal

from fastapi import HTTPException

from rl_fzerox.apps.run_manager.api.payloads.runs import run_payload
from rl_fzerox.core.manager import ManagedRun, ManagerStore
from rl_fzerox.core.manager.models import PolicySourceArtifact


def run_response(store: ManagerStore, run: ManagedRun) -> dict[str, dict[str, object]]:
    recent_events = store.list_recent_run_events((run.id,), limit_per_run=6)
    return {
        "run": run_payload(
            run,
            recent_events=recent_events.get(run.id, ()),
            active_alt_baseline_count=active_alt_baseline_count(store, run.id),
            available_policy_artifacts=_published_checkpoint_policy_artifacts(store, run),
        )
    }


def run_response_for_id(store: ManagerStore, run_id: str) -> dict[str, dict[str, object]]:
    return run_response(store, require_run(store, run_id))


def require_run(store: ManagerStore, run_id: str) -> ManagedRun:
    run = store.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="run not found")
    return run


def active_alt_baseline_count(store: ManagerStore, run_id: str) -> int:
    return len(store.active_run_alt_baselines(run_id))


def _published_checkpoint_policy_artifacts(
    store: ManagerStore,
    run: ManagedRun,
) -> tuple[PolicySourceArtifact, ...] | None:
    checkpoint = store.get_published_checkpoint_by_run_id(run.id)
    if checkpoint is None:
        return None
    if checkpoint.policy_path.is_file() and checkpoint.model_path.is_file():
        return (checkpoint.source_artifact,)
    return ()


def validate_source_fields(
    *,
    source_run_id: str | None,
    source_artifact: Literal["latest", "best"] | None,
) -> None:
    if source_run_id is None and source_artifact is None:
        return
    if source_run_id is None or source_artifact is None:
        raise HTTPException(
            status_code=400,
            detail="source_run_id and source_artifact must either both be set or both be null",
        )
