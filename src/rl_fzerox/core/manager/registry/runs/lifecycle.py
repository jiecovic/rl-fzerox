# src/rl_fzerox/core/manager/registry/runs/lifecycle.py
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from rl_fzerox.core.manager.models import (
    ManagedRun,
    ManagedRunEvent,
    RunStatus,
)
from rl_fzerox.core.manager.registry.common import (
    new_run_id,
    raise_name_conflict,
    sql_placeholders,
    utc_now,
)
from rl_fzerox.core.manager.registry.rows import run_from_row, run_select_sql
from rl_fzerox.core.manager.run_spec import ManagedRunConfig
from rl_fzerox.core.manager.storage.serialization import config_hash, config_json

if TYPE_CHECKING:
    from rl_fzerox.core.manager.store import ManagerStore


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

    del exclude_draft_id
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
    from rl_fzerox.core.manager.registry.runs.maintenance import reconcile_orphaned_runs

    reconcile_orphaned_runs(store)
    with store._connect() as connection:
        row = connection.execute(
            run_select_sql(where_clause="WHERE runs.id = ?"),
            (run_id,),
        ).fetchone()
    return None if row is None else run_from_row(row)


def list_runs(store: ManagerStore) -> tuple[ManagedRun, ...]:
    store.initialize()
    from rl_fzerox.core.manager.registry.runs.maintenance import reconcile_orphaned_runs

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


def update_run_name(
    store: ManagerStore,
    *,
    run_id: str,
    name: str,
) -> ManagedRun | None:
    store.initialize()
    normalized_name = name.strip() or run_id
    renamed_at = utc_now()
    try:
        with store._connect() as connection:
            row = connection.execute(
                """
                UPDATE runs
                SET name = ?
                WHERE id = ?
                RETURNING id
                """,
                (normalized_name, run_id),
            ).fetchone()
            if row is None:
                return None
            connection.execute(
                """
                INSERT INTO run_events(run_id, created_at, kind, message)
                VALUES (?, ?, ?, ?)
                """,
                (run_id, renamed_at, "renamed", f"run renamed to {normalized_name}"),
            )
            selected_row = connection.execute(
                run_select_sql(where_clause="WHERE runs.id = ?"),
                (run_id,),
            ).fetchone()
    except sqlite3.IntegrityError as error:
        raise_name_conflict(error, table="runs", kind="run", name=normalized_name)
        raise
    return None if selected_row is None else run_from_row(selected_row)
