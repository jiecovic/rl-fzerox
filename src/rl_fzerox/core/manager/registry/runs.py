# src/rl_fzerox/core/manager/registry/runs.py
from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import rl_fzerox.core.manager.filesystem_ops as filesystem_ops
from rl_fzerox.core.manager.config import ManagedRunConfig
from rl_fzerox.core.manager.models import (
    ManagedRun,
    ManagedRunEvent,
    RunCommand,
    RunStatus,
)
from rl_fzerox.core.manager.registry.common import (
    new_run_id,
    pid_exists,
    raise_name_conflict,
    run_command,
    sql_placeholders,
    utc_now,
)
from rl_fzerox.core.manager.registry.rows import (
    run_from_row,
    run_select_sql,
    run_worker_lease_from_row,
)
from rl_fzerox.core.manager.serialization import config_hash, config_json

if TYPE_CHECKING:
    from rl_fzerox.core.manager.store import ManagerStore


RUN_WORKER_HEARTBEAT_TIMEOUT = timedelta(seconds=90)


def create_run(
    store: ManagerStore,
    *,
    run_id: str | None = None,
    name: str,
    config: ManagedRunConfig,
    managed_runs_root: Path | None = None,
    explicit_run_dir: Path | None = None,
    lineage_id: str | None = None,
    lineage_step_offset: int = 0,
    parent_run_id: str | None = None,
    source_run_id: str | None = None,
    source_artifact: Literal["latest", "best"] | None = None,
    source_snapshot_dir: Path | None = None,
    source_num_timesteps: int | None = None,
    exclude_draft_id: str | None = None,
) -> ManagedRun:
    """Create an immutable SQLite run record without filesystem side effects."""

    from rl_fzerox.core.manager.registry.lineages import resolve_lineage_id
    from rl_fzerox.core.manager.registry.paths import manager_run_dir

    store.initialize()
    created_at = utc_now()
    resolved_run_id = run_id or new_run_id()
    normalized_name = name.strip() or resolved_run_id
    root = (managed_runs_root or store.manager_runs_root()).expanduser().resolve()
    run: ManagedRun | None = None

    try:
        with store._connect() as connection:
            resolved_lineage_id = resolve_lineage_id(
                connection,
                explicit_lineage_id=lineage_id,
                parent_run_id=parent_run_id,
                source_run_id=source_run_id,
                fallback_run_id=resolved_run_id,
            )
            run_dir = (
                (
                    explicit_run_dir
                    or manager_run_dir(
                        run_id=resolved_run_id,
                        lineage_id=resolved_lineage_id,
                        output_root=root,
                    )
                )
                .expanduser()
                .resolve()
            )
            run = ManagedRun(
                id=resolved_run_id,
                name=normalized_name,
                status="created",
                config=config,
                config_hash=config_hash(config),
                run_dir=run_dir,
                lineage_id=resolved_lineage_id,
                lineage_step_offset=lineage_step_offset,
                parent_run_id=parent_run_id,
                source_run_id=source_run_id,
                source_artifact=source_artifact,
                source_snapshot_dir=source_snapshot_dir,
                source_num_timesteps=source_num_timesteps,
                created_at=created_at,
            )
            connection.execute(
                """
                INSERT INTO runs(
                    id,
                    name,
                    status,
                    config_json,
                    config_hash,
                    run_dir,
                    lineage_id,
                    lineage_step_offset,
                    parent_run_id,
                    source_run_id,
                    source_artifact,
                    source_snapshot_dir,
                    source_num_timesteps,
                    created_at,
                    started_at,
                    stopped_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run.id,
                    run.name,
                    run.status,
                    config_json(run.config),
                    run.config_hash,
                    str(run.run_dir),
                    run.lineage_id,
                    run.lineage_step_offset,
                    run.parent_run_id,
                    run.source_run_id,
                    run.source_artifact,
                    None if run.source_snapshot_dir is None else str(run.source_snapshot_dir),
                    run.source_num_timesteps,
                    run.created_at,
                    run.started_at,
                    run.stopped_at,
                ),
            )
            connection.execute(
                """
                INSERT INTO run_events(run_id, created_at, kind, message)
                VALUES (?, ?, ?, ?)
                """,
                (run.id, created_at, "created", "run created from manager config"),
            )
    except sqlite3.IntegrityError as error:
        raise_name_conflict(error, table="runs", kind="run", name=normalized_name)
        raise
    if run is None:
        raise RuntimeError("managed run creation failed before insert")
    return run


def get_run(store: ManagerStore, run_id: str) -> ManagedRun | None:
    store.initialize()
    reconcile_orphaned_runs(store)
    with store._connect() as connection:
        row = connection.execute(
            run_select_sql(where_clause="WHERE runs.id = ?"),
            (run_id,),
        ).fetchone()
    return None if row is None else run_from_row(row)


def list_runs(store: ManagerStore) -> tuple[ManagedRun, ...]:
    store.initialize()
    reconcile_orphaned_runs(store)
    with store._connect() as connection:
        rows = connection.execute(
            run_select_sql(order_clause="ORDER BY runs.created_at DESC, runs.id DESC")
        ).fetchall()
    return tuple(run_from_row(row) for row in rows)


def list_visible_runs(store: ManagerStore) -> tuple[ManagedRun, ...]:
    return tuple(run for run in list_runs(store) if run.status != "created")


def list_recent_run_events(
    store: ManagerStore,
    run_ids: tuple[str, ...],
    *,
    limit_per_run: int = 6,
) -> dict[str, tuple[ManagedRunEvent, ...]]:
    if not run_ids:
        return {}
    store.initialize()
    with store._connect() as connection:
        rows = connection.execute(
            f"""
            SELECT run_id, created_at, kind, message
            FROM (
                SELECT
                    run_id,
                    id,
                    created_at,
                    kind,
                    message,
                    ROW_NUMBER() OVER (
                        PARTITION BY run_id
                        ORDER BY created_at DESC, id DESC
                    ) AS row_num
                FROM run_events
                WHERE run_id IN ({sql_placeholders(len(run_ids))})
            )
            WHERE row_num <= ?
            ORDER BY created_at DESC, id DESC
            """,
            (*run_ids, limit_per_run),
        ).fetchall()
    events_by_run_id: dict[str, list[ManagedRunEvent]] = {run_id: [] for run_id in run_ids}
    for row in rows:
        event_run_id = str(row["run_id"])
        events_by_run_id.setdefault(event_run_id, []).append(
            ManagedRunEvent(
                run_id=event_run_id,
                created_at=str(row["created_at"]),
                kind=str(row["kind"]),
                message=str(row["message"]),
            )
        )
    return {
        event_run_id: tuple(events) for event_run_id, events in events_by_run_id.items() if events
    }


def update_run_status(
    store: ManagerStore,
    *,
    run_id: str,
    status: RunStatus,
    message: str,
    started_at: str | None = None,
    stopped_at: str | None = None,
) -> ManagedRun | None:
    store.initialize()
    event_at = utc_now()
    with store._connect() as connection:
        row = connection.execute(
            """
            UPDATE runs
            SET
                status = ?,
                started_at = CASE
                    WHEN ? IS NULL THEN started_at
                    ELSE ?
                END,
                stopped_at = ?
            WHERE id = ?
            RETURNING id
            """,
            (
                status,
                started_at,
                started_at,
                stopped_at,
                run_id,
            ),
        ).fetchone()
        if row is None:
            return None
        if status != "running":
            connection.execute("DELETE FROM run_workers WHERE run_id = ?", (run_id,))
        connection.execute(
            """
            INSERT INTO run_events(run_id, created_at, kind, message)
            VALUES (?, ?, ?, ?)
            """,
            (run_id, event_at, status, message),
        )
        selected_row = connection.execute(
            run_select_sql(where_clause="WHERE runs.id = ?"),
            (run_id,),
        ).fetchone()
    return None if selected_row is None else run_from_row(selected_row)


def clear_run_runtime(store: ManagerStore, run_id: str) -> None:
    store._ensure_schema_initialized()
    with store._connect() as connection:
        connection.execute("DELETE FROM run_runtime WHERE run_id = ?", (run_id,))


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


def upsert_run_runtime(
    store: ManagerStore,
    *,
    run_id: str,
    total_timesteps: int,
    num_timesteps: int,
    progress_fraction: float,
    updated_at: str,
    fps: float | None = None,
    episode_reward_mean: float | None = None,
    episode_length_mean: float | None = None,
    approx_kl: float | None = None,
    entropy_loss: float | None = None,
    value_loss: float | None = None,
    policy_gradient_loss: float | None = None,
) -> None:
    store._ensure_schema_initialized()
    with store._connect() as connection:
        connection.execute(
            """
            INSERT INTO run_runtime(
                run_id,
                total_timesteps,
                num_timesteps,
                progress_fraction,
                updated_at,
                fps,
                episode_reward_mean,
                episode_length_mean,
                approx_kl,
                entropy_loss,
                value_loss,
                policy_gradient_loss
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_id) DO UPDATE SET
                total_timesteps = excluded.total_timesteps,
                num_timesteps = excluded.num_timesteps,
                progress_fraction = excluded.progress_fraction,
                updated_at = excluded.updated_at,
                fps = excluded.fps,
                episode_reward_mean = excluded.episode_reward_mean,
                episode_length_mean = excluded.episode_length_mean,
                approx_kl = excluded.approx_kl,
                entropy_loss = excluded.entropy_loss,
                value_loss = excluded.value_loss,
                policy_gradient_loss = excluded.policy_gradient_loss
            """,
            (
                run_id,
                total_timesteps,
                num_timesteps,
                progress_fraction,
                updated_at,
                fps,
                episode_reward_mean,
                episode_length_mean,
                approx_kl,
                entropy_loss,
                value_loss,
                policy_gradient_loss,
            ),
        )


def append_run_event(
    store: ManagerStore,
    *,
    run_id: str,
    kind: str,
    message: str,
    created_at: str | None = None,
) -> None:
    store._ensure_schema_initialized()
    event_at = created_at or utc_now()
    with store._connect() as connection:
        connection.execute(
            """
            INSERT INTO run_events(run_id, created_at, kind, message)
            VALUES (?, ?, ?, ?)
            """,
            (run_id, event_at, kind, message),
        )


def update_run_fork_source(
    store: ManagerStore,
    *,
    run_id: str,
    source_snapshot_dir: Path,
    source_num_timesteps: int,
    lineage_step_offset: int | None = None,
) -> ManagedRun | None:
    store.initialize()
    rebuilt_at = utc_now()
    with store._connect() as connection:
        row = connection.execute(
            """
            UPDATE runs
            SET
                source_snapshot_dir = ?,
                source_num_timesteps = ?,
                lineage_step_offset = COALESCE(?, lineage_step_offset)
            WHERE id = ?
            RETURNING id
            """,
            (
                str(source_snapshot_dir.expanduser().resolve()),
                source_num_timesteps,
                lineage_step_offset,
                run_id,
            ),
        ).fetchone()
        if row is None:
            return None
        connection.execute(
            """
            INSERT INTO run_events(run_id, created_at, kind, message)
            VALUES (?, ?, ?, ?)
            """,
            (
                run_id,
                rebuilt_at,
                "fork_source_rebuilt",
                "recreated pinned fork source snapshot",
            ),
        )
        selected_row = connection.execute(
            run_select_sql(where_clause="WHERE runs.id = ?"),
            (run_id,),
        ).fetchone()
    return None if selected_row is None else run_from_row(selected_row)


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
