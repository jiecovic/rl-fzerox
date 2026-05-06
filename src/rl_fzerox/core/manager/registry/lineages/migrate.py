# src/rl_fzerox/core/manager/registry/lineages/migrate.py
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING

from rl_fzerox.core.manager.artifacts.filesystem import queue_move_tree
from rl_fzerox.core.manager.registry import paths as registry_paths
from rl_fzerox.core.manager.registry.common import (
    optional_path,
    optional_str,
    utc_now,
)
from rl_fzerox.core.manager.registry.lineages.order import is_relative_to

if TYPE_CHECKING:
    from rl_fzerox.core.manager.store import ManagerStore


def migrate_lineage_layout(store: ManagerStore) -> int:
    store.initialize()
    with store._connect() as connection:
        backfill_lineage_ids(connection)
        moved = migrate_lineage_layout_rows(connection, migrated_at=utc_now())
    store._drain_pending_filesystem_operations()
    return moved


def resolve_lineage_id(
    connection: sqlite3.Connection,
    *,
    explicit_lineage_id: str | None,
    parent_run_id: str | None,
    source_run_id: str | None,
    fallback_run_id: str,
) -> str:
    if explicit_lineage_id is not None:
        return explicit_lineage_id
    parent_id = parent_run_id or source_run_id
    if parent_id is None:
        return fallback_run_id
    row = connection.execute(
        """
        SELECT lineage_id
        FROM runs
        WHERE id = ?
        """,
        (parent_id,),
    ).fetchone()
    if row is None:
        return fallback_run_id
    lineage_id = optional_str(row["lineage_id"])
    return lineage_id or parent_id


def backfill_lineage_ids(connection: sqlite3.Connection) -> None:
    rows = connection.execute(
        """
        SELECT id, lineage_id, parent_run_id, source_run_id
        FROM runs
        ORDER BY created_at ASC, id ASC
        """
    ).fetchall()
    if not rows:
        return
    row_by_id = {str(row["id"]): row for row in rows}
    resolved: dict[str, str] = {}

    def resolve(run_id: str) -> str:
        existing = resolved.get(run_id)
        if existing is not None:
            return existing
        row = row_by_id[run_id]
        current_value = optional_str(row["lineage_id"])
        if current_value is not None:
            resolved[run_id] = current_value
            return current_value
        parent_id = optional_str(row["parent_run_id"]) or optional_str(row["source_run_id"])
        lineage_id = (
            run_id if parent_id is None or parent_id not in row_by_id else resolve(parent_id)
        )
        resolved[run_id] = lineage_id
        return lineage_id

    for run_id in row_by_id:
        lineage_id = resolve(run_id)
        connection.execute(
            """
            UPDATE runs
            SET lineage_id = ?
            WHERE id = ? AND (lineage_id IS NULL OR lineage_id != ?)
            """,
            (lineage_id, run_id, lineage_id),
        )


def migrate_lineage_layout_rows(
    connection: sqlite3.Connection,
    *,
    migrated_at: str,
) -> int:
    moved = 0
    managed_runs_root = registry_paths.manager_root()
    legacy_lineages_root = managed_runs_root.parent / "lineages"
    rows = connection.execute(
        """
        SELECT id, lineage_id, run_dir, source_snapshot_dir
        FROM runs
        ORDER BY created_at ASC, id ASC
        """
    ).fetchall()
    for row in rows:
        run_id = str(row["id"])
        lineage_id = str(row["lineage_id"] or run_id)
        current_run_dir = Path(str(row["run_dir"])).expanduser().resolve()
        if not (
            is_relative_to(current_run_dir, managed_runs_root)
            or is_relative_to(current_run_dir, legacy_lineages_root)
        ):
            continue
        target_run_dir = registry_paths.manager_run_dir(run_id=run_id, lineage_id=lineage_id)
        current_snapshot_dir = optional_path(row["source_snapshot_dir"])
        target_snapshot_dir = current_snapshot_dir
        if current_snapshot_dir is not None and is_relative_to(
            current_snapshot_dir, current_run_dir
        ):
            target_snapshot_dir = target_run_dir / current_snapshot_dir.relative_to(current_run_dir)
        if current_run_dir == target_run_dir:
            if target_snapshot_dir != current_snapshot_dir:
                connection.execute(
                    """
                    UPDATE runs
                    SET source_snapshot_dir = ?
                    WHERE id = ?
                    """,
                    (
                        None if target_snapshot_dir is None else str(target_snapshot_dir),
                        run_id,
                    ),
                )
            continue
        if not current_run_dir.exists():
            continue
        queue_move_tree(
            connection,
            source_path=current_run_dir,
            target_path=target_run_dir,
            created_at=migrated_at,
        )
        connection.execute(
            """
            UPDATE runs
            SET run_dir = ?, source_snapshot_dir = ?
            WHERE id = ?
            """,
            (
                str(target_run_dir),
                None if target_snapshot_dir is None else str(target_snapshot_dir),
                run_id,
            ),
        )
        moved += 1
    return moved
