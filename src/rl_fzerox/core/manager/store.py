# src/rl_fzerox/core/manager/store.py
from __future__ import annotations

import os
import re
import shutil
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal
from uuid import uuid4

from rl_fzerox.core.manager.config import ManagedRunConfig
from rl_fzerox.core.manager.errors import ManagerNameConflictError
from rl_fzerox.core.manager.fork_source import (
    draft_fork_source_dir,
    reset_fork_source_dir,
    snapshot_fork_source,
)
from rl_fzerox.core.manager.models import (
    ManagedRun,
    ManagedRunDraft,
    ManagedRunEvent,
    ManagedRunRuntime,
    ManagedRunTemplate,
    RunCommand,
    RunStatus,
)
from rl_fzerox.core.manager.paths import manager_runs_root, predicted_managed_run_dir
from rl_fzerox.core.manager.schema import initialize_manager_schema
from rl_fzerox.core.manager.serialization import config_hash, config_json, load_config_json

RUN_WORKER_HEARTBEAT_TIMEOUT = timedelta(seconds=90)


def default_manager_db_path() -> Path:
    """Return the default local manager registry path."""

    return Path("local/manager/runs.db").resolve()


def new_managed_run_id(name: str) -> str:
    """Return one stable opaque id for a managed run."""

    del name
    return _new_run_id()


class ManagerStore:
    """SQLite-backed source of truth for managed training runs."""

    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = (db_path or default_manager_db_path()).expanduser().resolve()
        self._schema_initialized = False

    def initialize(self) -> None:
        """Create the manager database schema if needed."""

        self._initialize_schema()

    def _ensure_schema_initialized(self) -> None:
        """Initialize schema once for hot-path store operations."""

        if self._schema_initialized:
            return
        self._initialize_schema()

    def _initialize_schema(self) -> None:
        """Apply schema/bootstrap work to the manager database."""

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            initialize_manager_schema(connection, applied_at=_utc_now())
            _backfill_lineage_ids(connection)
            _migrate_lineage_layout(connection)
        self._schema_initialized = True

    def create_run(
        self,
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

        self.initialize()
        created_at = _utc_now()
        resolved_run_id = run_id or _new_run_id()
        normalized_name = name.strip() or resolved_run_id
        root = (managed_runs_root or manager_runs_root()).expanduser().resolve()
        run: ManagedRun | None = None

        try:
            with self._connect() as connection:
                resolved_lineage_id = _resolve_lineage_id(
                    connection,
                    explicit_lineage_id=lineage_id,
                    parent_run_id=parent_run_id,
                    source_run_id=source_run_id,
                    fallback_run_id=resolved_run_id,
                )
                run_dir = (
                    explicit_run_dir
                    or predicted_managed_run_dir(
                        resolved_run_id,
                        lineage_id=resolved_lineage_id,
                        output_root=root,
                    )
                ).expanduser().resolve()
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
            _raise_name_conflict(error, table="runs", kind="run", name=normalized_name)
            raise
        if run is None:
            raise RuntimeError("managed run creation failed before insert")
        return run

    def get_run(self, run_id: str) -> ManagedRun | None:
        """Return one DB-managed run by id."""

        self.initialize()
        self.reconcile_orphaned_runs()
        with self._connect() as connection:
            row = connection.execute(
                _run_select_sql(where_clause="WHERE runs.id = ?"),
                (run_id,),
            ).fetchone()
        return None if row is None else _run_from_row(row)

    def create_draft(
        self,
        *,
        name: str,
        config: ManagedRunConfig,
        source_run_id: str | None = None,
        source_artifact: Literal["latest", "best"] | None = None,
    ) -> ManagedRunDraft:
        """Persist one mutable draft and pin a fork source when requested."""

        self.initialize()
        created_at = _utc_now()
        draft_id = _new_record_id(name)
        normalized_name = name.strip() or draft_id
        source_snapshot_dir: Path | None = None
        source_num_timesteps: int | None = None
        source_run = None if source_run_id is None else self.get_run(source_run_id)
        if source_run_id is not None and source_run is None:
            raise ValueError(f"run not found: {source_run_id}")
        draft = ManagedRunDraft(
            id=draft_id,
            name=normalized_name,
            config=config,
            config_hash=config_hash(config),
            source_run_id=source_run_id,
            source_artifact=source_artifact,
            source_snapshot_dir=source_snapshot_dir,
            source_num_timesteps=source_num_timesteps,
            created_at=created_at,
            updated_at=created_at,
        )

        try:
            with self._connect() as connection:
                _assert_name_available(connection, normalized_name)
                if source_run_id is not None and source_artifact is not None:
                    assert source_run is not None
                    source_snapshot_dir = draft_fork_source_dir(
                        manager_db_path=self.db_path,
                        draft_id=draft_id,
                    )
                    source_num_timesteps = snapshot_fork_source(
                        source_run_dir=source_run.run_dir,
                        artifact=source_artifact,
                        destination_dir=source_snapshot_dir,
                    )
                    draft = ManagedRunDraft(
                        id=draft.id,
                        name=draft.name,
                        config=draft.config,
                        config_hash=draft.config_hash,
                        created_at=draft.created_at,
                        updated_at=draft.updated_at,
                        source_run_id=source_run_id,
                        source_artifact=source_artifact,
                        source_snapshot_dir=source_snapshot_dir,
                        source_num_timesteps=source_num_timesteps,
                    )
                connection.execute(
                    """
                    INSERT INTO run_drafts(
                        id,
                        name,
                        config_json,
                        config_hash,
                        source_run_id,
                        source_artifact,
                        source_snapshot_dir,
                        source_num_timesteps,
                        created_at,
                        updated_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        draft.id,
                        draft.name,
                        config_json(draft.config),
                        draft.config_hash,
                        draft.source_run_id,
                        draft.source_artifact,
                        (
                            None
                            if draft.source_snapshot_dir is None
                            else str(draft.source_snapshot_dir)
                        ),
                        draft.source_num_timesteps,
                        draft.created_at,
                        draft.updated_at,
                    ),
                )
        except sqlite3.IntegrityError as error:
            _raise_name_conflict(error, table="run_drafts", kind="draft", name=normalized_name)
            raise
        except Exception:
            if source_snapshot_dir is not None:
                reset_fork_source_dir(source_snapshot_dir)
            raise
        return draft

    def list_runs(self) -> tuple[ManagedRun, ...]:
        """Return all DB-managed runs, newest first."""

        self.initialize()
        self.reconcile_orphaned_runs()
        with self._connect() as connection:
            rows = connection.execute(
                _run_select_sql(order_clause="ORDER BY runs.created_at DESC, runs.id DESC")
            ).fetchall()
        return tuple(_run_from_row(row) for row in rows)

    def list_visible_runs(self) -> tuple[ManagedRun, ...]:
        """Return runs that should appear in the current run registry UI."""

        return tuple(run for run in self.list_runs() if run.status != "created")

    def list_recent_run_events(
        self,
        run_ids: tuple[str, ...],
        *,
        limit_per_run: int = 6,
    ) -> dict[str, tuple[ManagedRunEvent, ...]]:
        """Return recent manager events grouped by run id."""

        if not run_ids:
            return {}
        self.initialize()
        with self._connect() as connection:
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
                    WHERE run_id IN ({_sql_placeholders(len(run_ids))})
                )
                WHERE row_num <= ?
                ORDER BY created_at DESC, id DESC
                """,
                (*run_ids, limit_per_run),
            ).fetchall()

        events_by_run_id: dict[str, list[ManagedRunEvent]] = {run_id: [] for run_id in run_ids}
        for row in rows:
            run_id = str(row["run_id"])
            events_by_run_id.setdefault(run_id, []).append(
                ManagedRunEvent(
                    run_id=run_id,
                    created_at=str(row["created_at"]),
                    kind=str(row["kind"]),
                    message=str(row["message"]),
                )
            )
        return {
            run_id: tuple(events)
            for run_id, events in events_by_run_id.items()
            if events
        }

    def update_run_status(
        self,
        *,
        run_id: str,
        status: RunStatus,
        message: str,
        started_at: str | None = None,
        stopped_at: str | None = None,
    ) -> ManagedRun | None:
        """Update one managed run status and append an event."""

        self.initialize()
        event_at = _utc_now()
        with self._connect() as connection:
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
                RETURNING
                    id,
                    name,
                    status,
                    config_json,
                    config_hash,
                    run_dir,
                    lineage_step_offset,
                    parent_run_id,
                    source_run_id,
                    source_artifact,
                    source_num_timesteps,
                    created_at,
                    started_at,
                    stopped_at
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
                _run_select_sql(where_clause="WHERE runs.id = ?"),
                (run_id,),
            ).fetchone()
        return None if selected_row is None else _run_from_row(selected_row)

    def clear_run_runtime(self, run_id: str) -> None:
        """Remove the latest runtime snapshot for one run."""

        self._ensure_schema_initialized()
        with self._connect() as connection:
            connection.execute(
                "DELETE FROM run_runtime WHERE run_id = ?",
                (run_id,),
            )

    def register_run_worker(
        self,
        *,
        run_id: str,
        launch_token: str,
        pid: int,
        launched_at: str,
    ) -> bool:
        """Persist the currently owned manager worker lease for one run."""

        self._ensure_schema_initialized()
        with self._connect() as connection:
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
        self,
        *,
        run_id: str,
        launch_token: str,
        heartbeat_at: str,
    ) -> bool:
        """Refresh one owned manager worker lease heartbeat."""

        self._ensure_schema_initialized()
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE run_workers
                SET heartbeat_at = ?
                WHERE run_id = ? AND launch_token = ?
                """,
                (heartbeat_at, run_id, launch_token),
            )
        return cursor.rowcount > 0

    def clear_run_worker(self, run_id: str, *, launch_token: str | None = None) -> None:
        """Delete the current manager worker lease for one run."""

        self._ensure_schema_initialized()
        with self._connect() as connection:
            if launch_token is None:
                connection.execute(
                    "DELETE FROM run_workers WHERE run_id = ?",
                    (run_id,),
                )
            else:
                connection.execute(
                    "DELETE FROM run_workers WHERE run_id = ? AND launch_token = ?",
                    (run_id, launch_token),
                )

    def upsert_run_runtime(
        self,
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
        """Persist the latest runtime snapshot for one managed run."""

        self._ensure_schema_initialized()
        with self._connect() as connection:
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
        self,
        *,
        run_id: str,
        kind: str,
        message: str,
        created_at: str | None = None,
    ) -> None:
        """Append one manager event for a run without mutating run status."""

        self._ensure_schema_initialized()
        event_at = created_at or _utc_now()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO run_events(run_id, created_at, kind, message)
                VALUES (?, ?, ?, ?)
                """,
                (run_id, event_at, kind, message),
            )

    def update_run_fork_source(
        self,
        *,
        run_id: str,
        source_snapshot_dir: Path,
        source_num_timesteps: int,
        lineage_step_offset: int | None = None,
    ) -> ManagedRun | None:
        """Persist one rebuilt pinned fork source for a managed run."""

        self.initialize()
        rebuilt_at = _utc_now()
        with self._connect() as connection:
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
                _run_select_sql(where_clause="WHERE runs.id = ?"),
                (run_id,),
            ).fetchone()
        return None if selected_row is None else _run_from_row(selected_row)

    def request_run_command(
        self,
        *,
        run_id: str,
        command: RunCommand,
    ) -> ManagedRun | None:
        """Persist one pending runtime control command for a run."""

        self.initialize()
        requested_at = _utc_now()
        with self._connect() as connection:
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
                _run_select_sql(where_clause="WHERE runs.id = ?"),
                (run_id,),
            ).fetchone()
        return None if row is None else _run_from_row(row)

    def pending_run_command(self, run_id: str) -> RunCommand | None:
        """Return the current pending command for one run, if any."""

        self._ensure_schema_initialized()
        with self._connect() as connection:
            row = connection.execute(
                "SELECT command FROM run_commands WHERE run_id = ?",
                (run_id,),
            ).fetchone()
        return _run_command(None if row is None else row["command"])

    def clear_run_command(
        self,
        run_id: str,
        *,
        command: RunCommand | None = None,
    ) -> None:
        """Clear one pending runtime control command."""

        self._ensure_schema_initialized()
        with self._connect() as connection:
            if command is None:
                connection.execute(
                    "DELETE FROM run_commands WHERE run_id = ?",
                    (run_id,),
                )
            else:
                connection.execute(
                    "DELETE FROM run_commands WHERE run_id = ? AND command = ?",
                    (run_id, command),
                )

    def list_drafts(self) -> tuple[ManagedRunDraft, ...]:
        """Return all SQLite-only drafts, newest first."""

        self.initialize()
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    id,
                    name,
                    config_json,
                    config_hash,
                    source_run_id,
                    source_artifact,
                    source_snapshot_dir,
                    source_num_timesteps,
                    created_at,
                    updated_at
                FROM run_drafts
                ORDER BY updated_at DESC, id DESC
                """
            ).fetchall()
        return tuple(_draft_from_row(row) for row in rows)

    def delete_draft(self, draft_id: str) -> bool:
        """Delete one persisted draft and its pinned fork snapshot, if any."""

        self.initialize()
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT source_snapshot_dir
                FROM run_drafts
                WHERE id = ?
                """,
                (draft_id,),
            ).fetchone()
            cursor = connection.execute(
                "DELETE FROM run_drafts WHERE id = ?",
                (draft_id,),
            )
        if row is not None and isinstance(row["source_snapshot_dir"], str):
            reset_fork_source_dir(Path(str(row["source_snapshot_dir"])))
        return cursor.rowcount > 0

    def delete_run(self, run_id: str) -> bool:
        """Delete one managed leaf run, its runtime rows, and its run directory."""

        self.initialize()
        with self._connect() as connection:
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
                shutil.rmtree(run_dir)

            connection.execute("DELETE FROM run_runtime WHERE run_id = ?", (run_id,))
            connection.execute("DELETE FROM run_commands WHERE run_id = ?", (run_id,))
            connection.execute("DELETE FROM run_workers WHERE run_id = ?", (run_id,))
            connection.execute("DELETE FROM run_events WHERE run_id = ?", (run_id,))
            cursor = connection.execute("DELETE FROM runs WHERE id = ?", (run_id,))
        return cursor.rowcount > 0

    def delete_lineage(self, lineage_id: str) -> bool:
        """Delete one full lineage, including its runs and dependent fork drafts."""

        self.initialize()
        with self._connect() as connection:
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
                WHERE run_id IN ({_sql_placeholders(len(run_ids))})
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
                WHERE source_run_id IN ({_sql_placeholders(len(run_ids))})
                """,
                run_ids,
            ).fetchall()
            run_delete_order = _delete_order_for_lineage(rows)

            for row in rows:
                run_dir = Path(str(row["run_dir"])).expanduser().resolve()
                if run_dir.exists():
                    shutil.rmtree(run_dir)
            for row in draft_rows:
                snapshot_dir = row["source_snapshot_dir"]
                if isinstance(snapshot_dir, str):
                    reset_fork_source_dir(Path(snapshot_dir))

            if draft_rows:
                draft_ids = tuple(str(row["id"]) for row in draft_rows)
                connection.execute(
                    f"DELETE FROM run_drafts WHERE id IN ({_sql_placeholders(len(draft_ids))})",
                    draft_ids,
                )
            for run_id in run_delete_order:
                connection.execute("DELETE FROM run_runtime WHERE run_id = ?", (run_id,))
                connection.execute("DELETE FROM run_commands WHERE run_id = ?", (run_id,))
                connection.execute("DELETE FROM run_workers WHERE run_id = ?", (run_id,))
                connection.execute("DELETE FROM run_events WHERE run_id = ?", (run_id,))
                connection.execute("DELETE FROM runs WHERE id = ?", (run_id,))

        lineage_dir = predicted_managed_run_dir(run_ids[0], lineage_id=lineage_id).parent
        if lineage_dir.exists():
            shutil.rmtree(lineage_dir, ignore_errors=True)
        return True

    def migrate_lineage_layout(self) -> int:
        """Move manager-owned run directories into the lineage layout."""

        self.initialize()
        with self._connect() as connection:
            _backfill_lineage_ids(connection)
            return _migrate_lineage_layout(connection)

    def update_run_name(self, *, run_id: str, name: str) -> ManagedRun | None:
        """Rename one managed run without mutating its frozen config."""

        self.initialize()
        normalized_name = name.strip() or run_id
        renamed_at = _utc_now()
        try:
            with self._connect() as connection:
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
                    _run_select_sql(where_clause="WHERE runs.id = ?"),
                    (run_id,),
                ).fetchone()
        except sqlite3.IntegrityError as error:
            _raise_name_conflict(error, table="runs", kind="run", name=normalized_name)
            raise
        return None if selected_row is None else _run_from_row(selected_row)

    def update_draft(
        self,
        *,
        draft_id: str,
        name: str,
        config: ManagedRunConfig,
        source_run_id: str | None = None,
        source_artifact: Literal["latest", "best"] | None = None,
    ) -> ManagedRunDraft | None:
        """Update one SQLite-backed draft in place."""

        self.initialize()
        updated_at = _utc_now()
        normalized_name = name.strip() or draft_id
        current_draft = self.get_draft(draft_id)
        if current_draft is None:
            return None
        source_run = None if source_run_id is None else self.get_run(source_run_id)
        if source_run_id is not None and source_run is None:
            raise ValueError(f"run not found: {source_run_id}")
        next_snapshot_dir = current_draft.source_snapshot_dir
        next_source_num_timesteps = current_draft.source_num_timesteps
        source_changed = (
            current_draft.source_run_id != source_run_id
            or current_draft.source_artifact != source_artifact
        )
        if source_changed and current_draft.source_run_id is not None:
            raise ValueError(
                "changing a fork draft source is not supported; "
                "create a new fork draft"
            )
        try:
            with self._connect() as connection:
                _assert_name_available(connection, normalized_name, exclude_draft_id=draft_id)
                if source_changed:
                    next_snapshot_dir = None
                    next_source_num_timesteps = None
                    if source_run_id is not None and source_artifact is not None:
                        assert source_run is not None
                        next_snapshot_dir = draft_fork_source_dir(
                            manager_db_path=self.db_path,
                            draft_id=draft_id,
                        )
                        next_source_num_timesteps = snapshot_fork_source(
                            source_run_dir=source_run.run_dir,
                            artifact=source_artifact,
                            destination_dir=next_snapshot_dir,
                        )
                update_params = (
                    normalized_name,
                    config_json(config),
                    config_hash(config),
                    source_run_id,
                    source_artifact,
                    None if next_snapshot_dir is None else str(next_snapshot_dir),
                    next_source_num_timesteps,
                    updated_at,
                    draft_id,
                )
                row = connection.execute(
                    """
                    UPDATE run_drafts
                    SET
                        name = ?,
                        config_json = ?,
                        config_hash = ?,
                        source_run_id = ?,
                        source_artifact = ?,
                        source_snapshot_dir = ?,
                        source_num_timesteps = ?,
                        updated_at = ?
                    WHERE id = ?
                    RETURNING
                        id,
                        name,
                        config_json,
                        config_hash,
                        source_run_id,
                        source_artifact,
                        source_snapshot_dir,
                        source_num_timesteps,
                        created_at,
                        updated_at
                    """,
                    update_params,
                ).fetchone()
        except sqlite3.IntegrityError as error:
            _raise_name_conflict(error, table="run_drafts", kind="draft", name=normalized_name)
            raise
        return None if row is None else _draft_from_row(row)

    def get_draft(self, draft_id: str) -> ManagedRunDraft | None:
        """Return one persisted draft by id."""

        self.initialize()
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT
                    id,
                    name,
                    config_json,
                    config_hash,
                    source_run_id,
                    source_artifact,
                    source_snapshot_dir,
                    source_num_timesteps,
                    created_at,
                    updated_at
                FROM run_drafts
                WHERE id = ?
                """,
                (draft_id,),
            ).fetchone()
        return None if row is None else _draft_from_row(row)

    def list_templates(self) -> tuple[ManagedRunTemplate, ...]:
        """Return available DB-backed run templates."""

        self.initialize()
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    id,
                    name,
                    config_json,
                    config_hash,
                    created_at,
                    updated_at
                FROM run_templates
                ORDER BY name COLLATE NOCASE
                """
            ).fetchall()
        return tuple(_template_from_row(row) for row in rows)

    def default_template(self) -> ManagedRunTemplate:
        """Return the first manager template, seeding the DB if needed."""

        templates = self.list_templates()
        if not templates:
            raise RuntimeError("Manager DB did not initialize a default run template")
        return templates[0]

    def reconcile_orphaned_runs(self) -> None:
        """Mark stale owned worker leases as failed when the worker has disappeared."""

        self.initialize()
        now = datetime.now(UTC)
        with self._connect() as connection:
            rows = connection.execute(
                _run_select_sql(where_clause="WHERE runs.status = 'running'")
            ).fetchall()
            worker_rows = connection.execute(
                """
                SELECT run_id, launch_token, pid, launched_at, heartbeat_at
                FROM run_workers
                """
            ).fetchall()
            worker_by_run_id = {
                str(row["run_id"]): _run_worker_lease_from_row(row)
                for row in worker_rows
            }
            for row in rows:
                run = _run_from_row(row)
                worker = worker_by_run_id.get(run.id)
                if worker is None:
                    continue
                heartbeat_at = datetime.fromisoformat(worker.heartbeat_at)
                if now - heartbeat_at <= RUN_WORKER_HEARTBEAT_TIMEOUT:
                    continue
                if _pid_exists(worker.pid):
                    continue
                connection.execute("DELETE FROM run_commands WHERE run_id = ?", (run.id,))
                connection.execute("DELETE FROM run_workers WHERE run_id = ?", (run.id,))
                connection.execute(
                    """
                    UPDATE runs
                    SET status = ?, stopped_at = ?
                    WHERE id = ?
                    """,
                    ("failed", _utc_now(), run.id),
                )
                connection.execute(
                    """
                    INSERT INTO run_events(run_id, created_at, kind, message)
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        run.id,
                        _utc_now(),
                        "failed",
                        "manager worker disappeared before reporting a clean final state",
                    ),
                )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA busy_timeout = 30000")
        connection.execute("PRAGMA foreign_keys = ON")
        return connection


def _run_from_row(row: sqlite3.Row) -> ManagedRun:
    return ManagedRun(
        id=str(row["id"]),
        name=str(row["name"]),
        status=_run_status(row["status"]),
        config=load_config_json(str(row["config_json"])),
        config_hash=str(row["config_hash"]),
        run_dir=Path(str(row["run_dir"])),
        lineage_id=str(row["lineage_id"] or row["id"]),
        parent_run_id=_optional_str(row["parent_run_id"]),
        source_run_id=_optional_str(row["source_run_id"]),
        source_artifact=_optional_source_artifact(row["source_artifact"]),
        source_snapshot_dir=_optional_path(row["source_snapshot_dir"]),
        source_num_timesteps=_optional_int(row["source_num_timesteps"]),
        created_at=str(row["created_at"]),
        lineage_step_offset=int(row["lineage_step_offset"]),
        started_at=_optional_str(row["started_at"]),
        stopped_at=_optional_str(row["stopped_at"]),
        runtime=_runtime_from_row(row),
        pending_command=_run_command(row["pending_command"]),
    )


def _runtime_from_row(row: sqlite3.Row) -> ManagedRunRuntime | None:
    if not isinstance(row["runtime_updated_at"], str):
        return None
    return ManagedRunRuntime(
        total_timesteps=int(row["runtime_total_timesteps"]),
        num_timesteps=int(row["runtime_num_timesteps"]),
        progress_fraction=float(row["runtime_progress_fraction"]),
        updated_at=row["runtime_updated_at"],
        fps=_optional_float(row["runtime_fps"]),
        episode_reward_mean=_optional_float(row["runtime_episode_reward_mean"]),
        episode_length_mean=_optional_float(row["runtime_episode_length_mean"]),
        approx_kl=_optional_float(row["runtime_approx_kl"]),
        entropy_loss=_optional_float(row["runtime_entropy_loss"]),
        value_loss=_optional_float(row["runtime_value_loss"]),
        policy_gradient_loss=_optional_float(row["runtime_policy_gradient_loss"]),
    )


@dataclass(frozen=True, slots=True)
class _RunWorkerLease:
    run_id: str
    launch_token: str
    pid: int
    launched_at: str
    heartbeat_at: str


def _run_worker_lease_from_row(row: sqlite3.Row) -> _RunWorkerLease:
    return _RunWorkerLease(
        run_id=str(row["run_id"]),
        launch_token=str(row["launch_token"]),
        pid=int(row["pid"]),
        launched_at=str(row["launched_at"]),
        heartbeat_at=str(row["heartbeat_at"]),
    )


def _template_from_row(row: sqlite3.Row) -> ManagedRunTemplate:
    return ManagedRunTemplate(
        id=str(row["id"]),
        name=str(row["name"]),
        config=load_config_json(str(row["config_json"])),
        config_hash=str(row["config_hash"]),
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
    )


def _draft_from_row(row: sqlite3.Row) -> ManagedRunDraft:
    return ManagedRunDraft(
        id=str(row["id"]),
        name=str(row["name"]),
        config=load_config_json(str(row["config_json"])),
        config_hash=str(row["config_hash"]),
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
        source_run_id=_optional_str(row["source_run_id"]),
        source_artifact=_optional_source_artifact(row["source_artifact"]),
        source_snapshot_dir=_optional_path(row["source_snapshot_dir"]),
        source_num_timesteps=_optional_int(row["source_num_timesteps"]),
    )


def _run_status(value: object) -> RunStatus:
    match value:
        case "created":
            return "created"
        case "running":
            return "running"
        case "paused":
            return "paused"
        case "stopped":
            return "stopped"
        case "finished":
            return "finished"
        case "failed":
            return "failed"
    raise ValueError(f"Unsupported managed run status: {value!r}")


def _run_command(value: object) -> RunCommand | None:
    match value:
        case None:
            return None
        case "pause":
            return "pause"
        case "stop":
            return "stop"
    raise ValueError(f"Unsupported managed run command: {value!r}")


def _optional_str(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _optional_float(value: object) -> float | None:
    return float(value) if isinstance(value, int | float) else None


def _optional_int(value: object) -> int | None:
    return value if isinstance(value, int) else None


def _optional_source_artifact(value: object) -> Literal["latest", "best"] | None:
    if value is None:
        return None
    artifact = str(value)
    if artifact == "latest":
        return "latest"
    if artifact == "best":
        return "best"
    raise ValueError(f"Unsupported managed source artifact: {artifact!r}")


def _optional_path(value: object) -> Path | None:
    return Path(str(value)).expanduser().resolve() if isinstance(value, str) else None


def _resolve_lineage_id(
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
    lineage_id = _optional_str(row["lineage_id"])
    return lineage_id or parent_id


def _backfill_lineage_ids(connection: sqlite3.Connection) -> None:
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
        current_value = _optional_str(row["lineage_id"])
        if current_value is not None:
            resolved[run_id] = current_value
            return current_value
        parent_id = _optional_str(row["parent_run_id"]) or _optional_str(row["source_run_id"])
        lineage_id = (
            run_id
            if parent_id is None or parent_id not in row_by_id
            else resolve(parent_id)
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


def _migrate_lineage_layout(connection: sqlite3.Connection) -> int:
    moved = 0
    managed_runs_root = manager_runs_root()
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
            _is_relative_to(current_run_dir, managed_runs_root)
            or _is_relative_to(current_run_dir, legacy_lineages_root)
        ):
            continue
        target_run_dir = predicted_managed_run_dir(run_id, lineage_id=lineage_id)
        current_snapshot_dir = _optional_path(row["source_snapshot_dir"])
        target_snapshot_dir = current_snapshot_dir
        if current_snapshot_dir is not None and _is_relative_to(
            current_snapshot_dir,
            current_run_dir,
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
        target_run_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(current_run_dir), str(target_run_dir))
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


def _delete_order_for_lineage(rows: list[sqlite3.Row]) -> tuple[str, ...]:
    parent_by_id = {
        str(row["id"]): _optional_str(row["parent_run_id"]) or _optional_str(row["source_run_id"])
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


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _sql_placeholders(count: int) -> str:
    if count <= 0:
        raise ValueError("count must be positive")
    return ", ".join("?" for _ in range(count))


def _run_select_sql(
    *,
    where_clause: str = "",
    order_clause: str = "",
) -> str:
    return f"""
        SELECT
            runs.id,
            runs.name,
            runs.status,
            runs.config_json,
            runs.config_hash,
            runs.run_dir,
            runs.lineage_id,
            runs.lineage_step_offset,
            runs.parent_run_id,
            runs.source_run_id,
            runs.source_artifact,
            runs.source_snapshot_dir,
            runs.source_num_timesteps,
            runs.created_at,
            runs.started_at,
            runs.stopped_at,
            run_runtime.total_timesteps AS runtime_total_timesteps,
            run_runtime.num_timesteps AS runtime_num_timesteps,
            run_runtime.progress_fraction AS runtime_progress_fraction,
            run_runtime.updated_at AS runtime_updated_at,
            run_runtime.fps AS runtime_fps,
            run_runtime.episode_reward_mean AS runtime_episode_reward_mean,
            run_runtime.episode_length_mean AS runtime_episode_length_mean,
            run_runtime.approx_kl AS runtime_approx_kl,
            run_runtime.entropy_loss AS runtime_entropy_loss,
            run_runtime.value_loss AS runtime_value_loss,
            run_runtime.policy_gradient_loss AS runtime_policy_gradient_loss,
            run_commands.command AS pending_command
        FROM runs
        LEFT JOIN run_runtime
            ON run_runtime.run_id = runs.id
        LEFT JOIN run_commands
            ON run_commands.run_id = runs.id
        {where_clause}
        {order_clause}
    """


def _new_run_id() -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    return f"{timestamp}-{uuid4().hex[:8]}"


def _new_record_id(name: str) -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    slug = _slugify(name) or "run"
    return f"{timestamp}-{slug}-{uuid4().hex[:8]}"


def _raise_name_conflict(
    error: sqlite3.IntegrityError,
    *,
    table: str,
    kind: str,
    name: str,
) -> None:
    if f"UNIQUE constraint failed: {table}.name" in str(error):
        raise ManagerNameConflictError(kind=kind, name=name) from error


def _assert_name_available(
    connection: sqlite3.Connection,
    name: str,
    *,
    exclude_draft_id: str | None = None,
) -> None:
    row = connection.execute(
        """
        SELECT 'draft' AS source
        FROM run_drafts
        WHERE name = ? COLLATE NOCASE
          AND (? IS NULL OR id != ?)
        LIMIT 1
        """,
        (
            name,
            exclude_draft_id,
            exclude_draft_id,
        ),
    ).fetchone()
    if row is not None:
        raise ManagerNameConflictError(kind=str(row["source"]), name=name)


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.strip().lower())
    return slug.strip("-")[:48]


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _pid_exists(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True
