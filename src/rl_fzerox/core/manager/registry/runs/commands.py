# src/rl_fzerox/core/manager/registry/runs/commands.py
from __future__ import annotations

from typing import TYPE_CHECKING

from rl_fzerox.core.manager.models import ManagedRun, RunCommand
from rl_fzerox.core.manager.registry.common import run_command, utc_now
from rl_fzerox.core.manager.registry.rows import run_from_row, run_select_sql

if TYPE_CHECKING:
    from rl_fzerox.core.manager.store import ManagerStore


def request_run_command(
    store: ManagerStore,
    *,
    run_id: str,
    command: RunCommand,
) -> ManagedRun | None:
    store.initialize()
    requested_at = utc_now()
    with store._connect() as connection:
        run_row = connection.execute(
            "SELECT 1 FROM runs WHERE id = ?",
            (run_id,),
        ).fetchone()
        if run_row is None:
            return None
        connection.execute(
            """
            INSERT INTO run_commands(run_id, command, requested_at)
            VALUES (?, ?, ?)
            ON CONFLICT(run_id) DO UPDATE SET
                command = excluded.command,
                requested_at = excluded.requested_at
            """,
            (run_id, command, requested_at),
        )
        connection.execute(
            """
            INSERT INTO run_events(run_id, created_at, kind, message)
            VALUES (?, ?, ?, ?)
            """,
            (run_id, requested_at, f"{command}_requested", f"{command} requested from manager"),
        )
        row = connection.execute(
            run_select_sql(where_clause="WHERE runs.id = ?"),
            (run_id,),
        ).fetchone()
    return None if row is None else run_from_row(row)


def pending_run_command(store: ManagerStore, run_id: str) -> RunCommand | None:
    store._ensure_schema_initialized()
    with store._connect() as connection:
        row = connection.execute(
            "SELECT command FROM run_commands WHERE run_id = ?",
            (run_id,),
        ).fetchone()
    return run_command(None if row is None else row["command"])


def clear_run_command(
    store: ManagerStore,
    run_id: str,
    *,
    command: RunCommand | None = None,
) -> None:
    store._ensure_schema_initialized()
    with store._connect() as connection:
        if command is None:
            connection.execute("DELETE FROM run_commands WHERE run_id = ?", (run_id,))
        else:
            connection.execute(
                "DELETE FROM run_commands WHERE run_id = ? AND command = ?",
                (run_id, command),
            )
