# src/rl_fzerox/core/manager/registry/runs/maintenance.py
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import rl_fzerox.core.manager.artifacts.filesystem as filesystem_ops
from rl_fzerox.core.manager.registry.common import pid_exists, utc_now
from rl_fzerox.core.manager.registry.rows import (
    run_from_row,
    run_select_sql,
    run_worker_lease_from_row,
)

if TYPE_CHECKING:
    from rl_fzerox.core.manager.store import ManagerStore


RUN_WORKER_HEARTBEAT_TIMEOUT = timedelta(seconds=90)


def reconcile_orphaned_runs(store: ManagerStore) -> None:
    store.initialize()
    now = datetime.now(UTC)
    with store._connect() as connection:
        rows = connection.execute(
            run_select_sql(where_clause="WHERE runs.status = 'running'")
        ).fetchall()
        worker_rows = connection.execute(
            """
            SELECT run_id, launch_token, pid, launched_at, heartbeat_at
            FROM run_workers
            """
        ).fetchall()
        worker_by_run_id = {
            str(row["run_id"]): run_worker_lease_from_row(row) for row in worker_rows
        }
        for row in rows:
            run = run_from_row(row)
            worker = worker_by_run_id.get(run.id)
            if worker is None:
                continue
            heartbeat_at = datetime.fromisoformat(worker.heartbeat_at)
            if now - heartbeat_at <= RUN_WORKER_HEARTBEAT_TIMEOUT:
                continue
            if pid_exists(worker.pid):
                continue
            connection.execute("DELETE FROM run_commands WHERE run_id = ?", (run.id,))
            connection.execute("DELETE FROM run_workers WHERE run_id = ?", (run.id,))
            connection.execute(
                """
                UPDATE runs
                SET status = ?, stopped_at = ?
                WHERE id = ?
                """,
                ("failed", utc_now(), run.id),
            )
            connection.execute(
                """
                INSERT INTO run_events(run_id, created_at, kind, message)
                VALUES (?, ?, ?, ?)
                """,
                (
                    run.id,
                    utc_now(),
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
