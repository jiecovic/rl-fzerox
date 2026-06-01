# src/rl_fzerox/core/manager/registry/runs/workers.py
from __future__ import annotations

from typing import TYPE_CHECKING

from rl_fzerox.core.manager.db.models import RunModel, RunWorkerModel

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
