# src/rl_fzerox/core/manager/registry/runs/maintenance.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from sqlalchemy import select

import rl_fzerox.core.manager.artifacts.filesystem as filesystem_ops
from rl_fzerox.core.manager.db.models import (
    RunCommandModel,
    RunEventModel,
    RunModel,
    RunWorkerModel,
)
from rl_fzerox.core.manager.db.repositories.filesystem import (
    delete_filesystem_operation,
    list_filesystem_operations,
)
from rl_fzerox.core.manager.registry.common import pid_exists, utc_now

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

    from rl_fzerox.core.manager.store import ManagerStore


@dataclass(frozen=True, slots=True)
class RunWorkerLeasePolicy:
    heartbeat_interval: timedelta = timedelta(seconds=3)
    heartbeat_timeout: timedelta = timedelta(seconds=90)


@dataclass(frozen=True, slots=True)
class RunWorkerLease:
    run_id: str
    launch_token: str
    pid: int
    launched_at: str
    heartbeat_at: str


RUN_WORKER_LEASE_POLICY = RunWorkerLeasePolicy()


def reconcile_orphaned_runs(store: ManagerStore) -> None:
    store.initialize()
    now = datetime.now(UTC)
    with store._orm_session() as session:
        run_ids = tuple(
            session.scalars(select(RunModel.id).where(RunModel.status == "running"))
        )
        worker_by_run_id = {
            worker.run_id: _run_worker_lease_from_model(worker)
            for worker in session.scalars(select(RunWorkerModel))
        }
        for run_id in run_ids:
            worker = worker_by_run_id.get(run_id)
            if worker is None:
                continue
            heartbeat_at = datetime.fromisoformat(worker.heartbeat_at)
            if now - heartbeat_at <= RUN_WORKER_LEASE_POLICY.heartbeat_timeout:
                continue
            if not pid_exists(worker.pid):
                _mark_orphaned_run_failed(session, run_id=run_id)
                continue


def _run_worker_lease_from_model(worker: RunWorkerModel) -> RunWorkerLease:
    return RunWorkerLease(
        run_id=worker.run_id,
        launch_token=worker.launch_token,
        pid=worker.pid,
        launched_at=worker.launched_at,
        heartbeat_at=worker.heartbeat_at,
    )


def _mark_orphaned_run_failed(session: Session, *, run_id: str) -> None:
    failed_at = utc_now()
    pending_command = session.get(RunCommandModel, run_id)
    if pending_command is not None:
        session.delete(pending_command)
    worker = session.get(RunWorkerModel, run_id)
    if worker is not None:
        session.delete(worker)
    run = session.get(RunModel, run_id)
    if run is None:
        return
    run.status = "failed"
    run.stopped_at = failed_at
    session.add(
        RunEventModel(
            run_id=run_id,
            created_at=failed_at,
            kind="failed",
            message="manager worker disappeared before reporting a clean final state",
        )
    )


def drain_pending_filesystem_operations(store: ManagerStore) -> None:
    store._ensure_schema_initialized()
    with store._orm_session() as session:
        operations = list_filesystem_operations(session)
    for operation in operations:
        try:
            complete = filesystem_ops.apply_filesystem_operation(operation)
        except Exception:
            if operation.kind == "move_tree":
                raise
            continue
        if not complete:
            continue
        with store._orm_session() as session:
            delete_filesystem_operation(session, operation.id)
