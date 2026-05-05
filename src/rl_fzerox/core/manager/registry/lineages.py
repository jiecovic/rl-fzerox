# src/rl_fzerox/core/manager/registry/lineages.py
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING

from rl_fzerox.core.manager.filesystem_ops import queue_delete_tree, queue_move_tree
from rl_fzerox.core.manager.registry import paths as registry_paths
from rl_fzerox.core.manager.registry.common import (
    optional_path,
    optional_str,
    sql_placeholders,
    utc_now,
)

if TYPE_CHECKING:
    from rl_fzerox.core.manager.store import ManagerStore


def delete_run(store: ManagerStore, run_id: str) -> bool:
    """Delete one managed leaf run and queue its filesystem cleanup."""

    store.initialize()
    deleted = False
    deleted_at = utc_now()
    with store._connect() as connection:
        row = connection.execute(
            """
            SELECT status, run_dir
            FROM runs
            WHERE id = ?
            """,
            (run_id,),
        ).fetchone()
        if row is None:
            return False
        if row["status"] == "running":
            raise ValueError("stop or pause the run before deleting it")
        if connection.execute(
            "SELECT 1 FROM run_commands WHERE run_id = ?",
            (run_id,),
        ).fetchone():
            raise ValueError("wait for the pending run command to finish before deleting it")
        if connection.execute(
            """
            SELECT 1
            FROM runs
            WHERE parent_run_id = ? OR source_run_id = ?
            LIMIT 1
            """,
            (run_id, run_id),
        ).fetchone():
            raise ValueError("only leaf runs can be deleted individually")
        if connection.execute(
            """
            SELECT 1
            FROM run_drafts
            WHERE source_run_id = ?
            LIMIT 1
            """,
            (run_id,),
        ).fetchone():
            raise ValueError("delete or retarget fork drafts that still depend on this run")

        run_dir = Path(str(row["run_dir"])).expanduser().resolve()
        if run_dir.exists():
            queue_delete_tree(connection, path=run_dir, created_at=deleted_at)

        connection.execute("DELETE FROM run_runtime WHERE run_id = ?", (run_id,))
        connection.execute("DELETE FROM run_commands WHERE run_id = ?", (run_id,))
        connection.execute("DELETE FROM run_workers WHERE run_id = ?", (run_id,))
        connection.execute("DELETE FROM run_events WHERE run_id = ?", (run_id,))
        cursor = connection.execute("DELETE FROM runs WHERE id = ?", (run_id,))
        deleted = cursor.rowcount > 0
    store._drain_pending_filesystem_operations()
    return deleted


def delete_lineage(store: ManagerStore, lineage_id: str) -> bool:
    """Delete one full lineage, including its runs and dependent fork drafts."""

    store.initialize()
    deleted = False
    deleted_at = utc_now()
    with store._connect() as connection:
        rows = connection.execute(
            """
            SELECT id, status, run_dir, parent_run_id, source_run_id
            FROM runs
            WHERE lineage_id = ?
            ORDER BY created_at DESC, id DESC
            """,
            (lineage_id,),
        ).fetchall()
        if not rows:
            return False
        run_ids = tuple(str(row["id"]) for row in rows)
        for row in rows:
            if row["status"] == "running":
                raise ValueError("stop all runs in this lineage before deleting it")
        pending = connection.execute(
            f"""
            SELECT 1
            FROM run_commands
            WHERE run_id IN ({sql_placeholders(len(run_ids))})
            LIMIT 1
            """,
            run_ids,
        ).fetchone()
        if pending is not None:
            raise ValueError("wait for pending run commands to finish before deleting lineage")

        draft_rows = connection.execute(
            f"""
            SELECT id, source_snapshot_dir
            FROM run_drafts
            WHERE source_run_id IN ({sql_placeholders(len(run_ids))})
            """,
            run_ids,
        ).fetchall()
        run_delete_order = delete_order_for_lineage(rows)
        lineage_dir = Path(str(rows[0]["run_dir"])).expanduser().resolve().parent

        for row in rows:
            run_dir = Path(str(row["run_dir"])).expanduser().resolve()
            if run_dir.exists():
                queue_delete_tree(connection, path=run_dir, created_at=deleted_at)
        for row in draft_rows:
            snapshot_dir = row["source_snapshot_dir"]
            if isinstance(snapshot_dir, str):
                queue_delete_tree(
                    connection,
                    path=Path(snapshot_dir),
                    created_at=deleted_at,
                )
        if lineage_dir.exists():
            queue_delete_tree(connection, path=lineage_dir, created_at=deleted_at)

        if draft_rows:
            draft_ids = tuple(str(row["id"]) for row in draft_rows)
            connection.execute(
                f"DELETE FROM run_drafts WHERE id IN ({sql_placeholders(len(draft_ids))})",
                draft_ids,
            )
        for current_run_id in run_delete_order:
            connection.execute("DELETE FROM run_runtime WHERE run_id = ?", (current_run_id,))
            connection.execute("DELETE FROM run_commands WHERE run_id = ?", (current_run_id,))
            connection.execute("DELETE FROM run_workers WHERE run_id = ?", (current_run_id,))
            connection.execute("DELETE FROM run_events WHERE run_id = ?", (current_run_id,))
            connection.execute("DELETE FROM runs WHERE id = ?", (current_run_id,))
        deleted = True
    store._drain_pending_filesystem_operations()
    return deleted


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


def delete_order_for_lineage(rows: list[sqlite3.Row]) -> tuple[str, ...]:
    parent_by_id = {
        str(row["id"]): optional_str(row["parent_run_id"]) or optional_str(row["source_run_id"])
        for row in rows
    }
    depth_by_id: dict[str, int] = {}

    def depth(run_id: str) -> int:
        existing = depth_by_id.get(run_id)
        if existing is not None:
            return existing
        parent_id = parent_by_id[run_id]
        if parent_id is None or parent_id not in parent_by_id:
            depth_by_id[run_id] = 0
            return 0
        resolved_depth = depth(parent_id) + 1
        depth_by_id[run_id] = resolved_depth
        return resolved_depth

    ordered = sorted(
        (str(row["id"]) for row in rows),
        key=lambda run_id: (depth(run_id), run_id),
        reverse=True,
    )
    return tuple(ordered)


def is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True
