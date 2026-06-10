# src/rl_fzerox/core/manager/registry/runs/workers.py
from __future__ import annotations

from typing import TYPE_CHECKING

from rl_fzerox.core.manager.db.models import (
    RunCommandModel,
    RunEventModel,
    RunModel,
    RunWorkerModel,
)

if TYPE_CHECKING:
    from rl_fzerox.core.manager.store import ManagerStore


def register_run_worker(
    store: ManagerStore,
    *,
    run_id: str,
    launch_token: str,
    pid: int,
    launched_at: str,
) -> bool:
    store._ensure_schema_initialized()
    with store._orm_session() as session:
        if session.get(RunModel, run_id) is None:
            return False
        worker = session.get(RunWorkerModel, run_id)
        if worker is None:
            session.add(
                RunWorkerModel(
                    run_id=run_id,
                    launch_token=launch_token,
                    pid=pid,
                    launched_at=launched_at,
                    heartbeat_at=launched_at,
                )
            )
        else:
            worker.launch_token = launch_token
            worker.pid = pid
            worker.launched_at = launched_at
            worker.heartbeat_at = launched_at
    return True


def heartbeat_run_worker(
    store: ManagerStore,
    *,
    run_id: str,
    launch_token: str,
    heartbeat_at: str,
) -> bool:
    store._ensure_schema_initialized()
    with store._orm_session() as session:
        worker = session.get(RunWorkerModel, run_id)
        if worker is None or worker.launch_token != launch_token:
            return False
        worker.heartbeat_at = heartbeat_at
    return True


def clear_run_worker(
    store: ManagerStore,
    run_id: str,
    *,
    launch_token: str | None = None,
) -> None:
    store._ensure_schema_initialized()
    with store._orm_session() as session:
        worker = session.get(RunWorkerModel, run_id)
        if worker is None:
            return
        if launch_token is None or worker.launch_token == launch_token:
            session.delete(worker)


def mark_worker_boot_failure(
    store: ManagerStore,
    *,
    run_id: str,
    launch_token: str,
    message: str,
    failed_at: str,
) -> bool:
    """Record failures that happen before the worker can load the run config."""

    store._ensure_schema_initialized()
    with store._orm_session() as session:
        run = session.get(RunModel, run_id)
        if run is None:
            return False
        pending = session.get(RunCommandModel, run_id)
        if pending is not None:
            session.delete(pending)
        worker = session.get(RunWorkerModel, run_id)
        if worker is not None and worker.launch_token == launch_token:
            session.delete(worker)
        run.status = "failed"
        run.stopped_at = failed_at
        session.add(
            RunEventModel(
                run_id=run_id,
                created_at=failed_at,
                kind="failed",
                message=message,
            )
        )
        return True
