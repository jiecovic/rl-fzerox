# src/rl_fzerox/core/manager/registry/lineages/delete.py
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from rl_fzerox.core.manager.artifacts.filesystem import queue_delete_tree
from rl_fzerox.core.manager.registry.common import sql_placeholders, utc_now
from rl_fzerox.core.manager.registry.lineages.order import delete_order_for_lineage

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

        connection.execute("DELETE FROM run_track_sampling_state WHERE run_id = ?", (run_id,))
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
        connection.execute("DELETE FROM lineage_groups WHERE lineage_id = ?", (lineage_id,))
        for current_run_id in run_delete_order:
            connection.execute(
                "DELETE FROM run_track_sampling_state WHERE run_id = ?",
                (current_run_id,),
            )
            connection.execute("DELETE FROM run_runtime WHERE run_id = ?", (current_run_id,))
            connection.execute("DELETE FROM run_commands WHERE run_id = ?", (current_run_id,))
            connection.execute("DELETE FROM run_workers WHERE run_id = ?", (current_run_id,))
            connection.execute("DELETE FROM run_events WHERE run_id = ?", (current_run_id,))
            connection.execute("DELETE FROM runs WHERE id = ?", (current_run_id,))
    store._drain_pending_filesystem_operations()
    return True
