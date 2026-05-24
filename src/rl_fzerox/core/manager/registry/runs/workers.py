# src/rl_fzerox/core/manager/registry/runs/workers.py
from __future__ import annotations

from typing import TYPE_CHECKING

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
    with store._connect() as connection:
        row = connection.execute(
            "SELECT 1 FROM runs WHERE id = ?",
            (run_id,),
        ).fetchone()
        if row is None:
            return False
        connection.execute(
            """
            INSERT INTO run_workers(
                run_id,
                launch_token,
                pid,
                launched_at,
                heartbeat_at
            )
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(run_id) DO UPDATE SET
                launch_token = excluded.launch_token,
                pid = excluded.pid,
                launched_at = excluded.launched_at,
                heartbeat_at = excluded.heartbeat_at
            """,
            (
                run_id,
                launch_token,
                pid,
                launched_at,
                launched_at,
            ),
        )
    return True


def heartbeat_run_worker(
    store: ManagerStore,
    *,
    run_id: str,
    launch_token: str,
    heartbeat_at: str,
) -> bool:
    store._ensure_schema_initialized()
    with store._connect() as connection:
        cursor = connection.execute(
            """
            UPDATE run_workers
            SET heartbeat_at = ?
            WHERE run_id = ? AND launch_token = ?
            """,
            (heartbeat_at, run_id, launch_token),
        )
    return cursor.rowcount > 0


def clear_run_worker(
    store: ManagerStore,
    run_id: str,
    *,
    launch_token: str | None = None,
) -> None:
    store._ensure_schema_initialized()
    with store._connect() as connection:
        if launch_token is None:
            connection.execute("DELETE FROM run_workers WHERE run_id = ?", (run_id,))
        else:
            connection.execute(
                "DELETE FROM run_workers WHERE run_id = ? AND launch_token = ?",
                (run_id, launch_token),
            )
