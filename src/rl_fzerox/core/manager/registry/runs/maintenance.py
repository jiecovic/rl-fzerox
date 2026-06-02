# src/rl_fzerox/core/manager/registry/runs/maintenance.py
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import rl_fzerox.core.manager.artifacts.filesystem as filesystem_ops
from rl_fzerox.core.manager.registry.common import pid_exists, utc_now

if TYPE_CHECKING:
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
    with store._connect() as connection:
        rows = connection.execute(
            """
            SELECT id
            FROM runs
            WHERE status = 'running'
            """
        ).fetchall()
        worker_rows = connection.execute(
            """
            SELECT run_id, launch_token, pid, launched_at, heartbeat_at
            FROM run_workers
            """
        ).fetchall()
        worker_by_run_id = {
            str(row["run_id"]): _run_worker_lease_from_row(row) for row in worker_rows
        }
        for row in rows:
            run_id = str(row["id"])
            worker = worker_by_run_id.get(run_id)
            if worker is None:
                continue
            heartbeat_at = datetime.fromisoformat(worker.heartbeat_at)
            if now - heartbeat_at <= RUN_WORKER_LEASE_POLICY.heartbeat_timeout:
                continue
            if not pid_exists(worker.pid):
                _mark_orphaned_run_failed(connection, run_id=run_id)
                continue


def _run_worker_lease_from_row(row: sqlite3.Row) -> RunWorkerLease:
    return RunWorkerLease(
        run_id=str(row["run_id"]),
        launch_token=str(row["launch_token"]),
        pid=int(row["pid"]),
        launched_at=str(row["launched_at"]),
        heartbeat_at=str(row["heartbeat_at"]),
    )


def _mark_orphaned_run_failed(connection: sqlite3.Connection, *, run_id: str) -> None:
    failed_at = utc_now()
    connection.execute("DELETE FROM run_commands WHERE run_id = ?", (run_id,))
    connection.execute("DELETE FROM run_workers WHERE run_id = ?", (run_id,))
    connection.execute(
        """
        UPDATE runs
        SET status = ?, stopped_at = ?
        WHERE id = ?
        """,
        ("failed", failed_at, run_id),
    )
    connection.execute(
        """
        INSERT INTO run_events(run_id, created_at, kind, message)
        VALUES (?, ?, ?, ?)
        """,
        (
            run_id,
            failed_at,
            "failed",
            "manager worker disappeared before reporting a clean final state",
        ),
    )


def drain_pending_filesystem_operations(store: ManagerStore) -> None:
    store._ensure_schema_initialized()
    with store._connect() as connection:
        rows = connection.execute(
            """
            SELECT id, kind, source_path, target_path, created_at
            FROM filesystem_operations
            ORDER BY id ASC
            """
        ).fetchall()
    for row in rows:
        operation = filesystem_ops.filesystem_operation_from_row(row)
        try:
            complete = filesystem_ops.apply_filesystem_operation(operation)
        except Exception:
            if operation.kind == "move_tree":
                raise
            continue
        if not complete:
            continue
        with store._connect() as connection:
            connection.execute(
                "DELETE FROM filesystem_operations WHERE id = ?",
                (operation.id,),
            )
