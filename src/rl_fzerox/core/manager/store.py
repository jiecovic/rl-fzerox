# src/rl_fzerox/core/manager/store.py
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Literal

from rl_fzerox.core.manager.config import ManagedRunConfig
from rl_fzerox.core.manager.models import (
    ManagedRun,
    ManagedRunDraft,
    ManagedRunEvent,
    ManagedRunTemplate,
    RunCommand,
    RunStatus,
)
from rl_fzerox.core.manager.paths import manager_runs_root
from rl_fzerox.core.manager.registry import common, rows
from rl_fzerox.core.manager.registry.common import new_run_id
from rl_fzerox.core.manager.registry.drafts import (
    create_draft,
    default_template,
    delete_draft,
    get_draft,
    list_drafts,
    list_templates,
    update_draft,
)
from rl_fzerox.core.manager.registry.lineages import (
    backfill_lineage_ids,
    delete_lineage,
    delete_run,
    migrate_lineage_layout,
    migrate_lineage_layout_rows,
)
from rl_fzerox.core.manager.registry.runs import (
    append_run_event,
    clear_run_command,
    clear_run_runtime,
    clear_run_worker,
    create_run,
    drain_pending_filesystem_operations,
    get_run,
    heartbeat_run_worker,
    list_recent_run_events,
    list_runs,
    list_visible_runs,
    pending_run_command,
    reconcile_orphaned_runs,
    register_run_worker,
    request_run_command,
    update_run_fork_source,
    update_run_status,
    upsert_run_runtime,
)
from rl_fzerox.core.manager.schema import initialize_manager_schema


def default_manager_db_path() -> Path:
    """Return the default local manager registry path."""

    return Path("local/manager/runs.db").resolve()


def new_managed_run_id(name: str) -> str:
    """Return one stable opaque id for a managed run."""

    del name
    return new_run_id()


class ManagerStore:
    """SQLite-backed source of truth for managed training runs."""

    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = (db_path or default_manager_db_path()).expanduser().resolve()
        self._schema_initialized = False

    def initialize(self) -> None:
        """Create the manager database schema if needed."""

        self._initialize_schema()
        self._drain_pending_filesystem_operations()

    def _ensure_schema_initialized(self) -> None:
        """Initialize schema once for hot-path store operations."""

        if self._schema_initialized:
            return
        self._initialize_schema()

    def _initialize_schema(self) -> None:
        """Apply schema/bootstrap work to the manager database."""

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            initialize_manager_schema(connection, applied_at=self.utc_now())
            backfill_lineage_ids(connection)
            migrate_lineage_layout_rows(connection, migrated_at=self.utc_now())
        self._schema_initialized = True

    def manager_runs_root(self, *, output_root: Path | None = None) -> Path:
        return manager_runs_root(output_root=output_root)

    def path(self, value: str | Path) -> Path:
        return Path(value).expanduser().resolve()

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
        return create_run(
            self,
            run_id=run_id,
            name=name,
            config=config,
            managed_runs_root=managed_runs_root,
            explicit_run_dir=explicit_run_dir,
            lineage_id=lineage_id,
            lineage_step_offset=lineage_step_offset,
            parent_run_id=parent_run_id,
            source_run_id=source_run_id,
            source_artifact=source_artifact,
            source_snapshot_dir=source_snapshot_dir,
            source_num_timesteps=source_num_timesteps,
            exclude_draft_id=exclude_draft_id,
        )

    def get_run(self, run_id: str) -> ManagedRun | None:
        return get_run(self, run_id)

    def create_draft(
        self,
        *,
        name: str,
        config: ManagedRunConfig,
        source_run_id: str | None = None,
        source_artifact: Literal["latest", "best"] | None = None,
    ) -> ManagedRunDraft:
        return create_draft(
            self,
            name=name,
            config=config,
            source_run_id=source_run_id,
            source_artifact=source_artifact,
        )

    def list_runs(self) -> tuple[ManagedRun, ...]:
        return list_runs(self)

    def list_visible_runs(self) -> tuple[ManagedRun, ...]:
        return list_visible_runs(self)

    def list_recent_run_events(
        self,
        run_ids: tuple[str, ...],
        *,
        limit_per_run: int = 6,
    ) -> dict[str, tuple[ManagedRunEvent, ...]]:
        return list_recent_run_events(self, run_ids, limit_per_run=limit_per_run)

    def update_run_status(
        self,
        *,
        run_id: str,
        status: RunStatus,
        message: str,
        started_at: str | None = None,
        stopped_at: str | None = None,
    ) -> ManagedRun | None:
        return update_run_status(
            self,
            run_id=run_id,
            status=status,
            message=message,
            started_at=started_at,
            stopped_at=stopped_at,
        )

    def clear_run_runtime(self, run_id: str) -> None:
        clear_run_runtime(self, run_id)

    def register_run_worker(
        self,
        *,
        run_id: str,
        launch_token: str,
        pid: int,
        launched_at: str,
    ) -> bool:
        return register_run_worker(
            self,
            run_id=run_id,
            launch_token=launch_token,
            pid=pid,
            launched_at=launched_at,
        )

    def heartbeat_run_worker(
        self,
        *,
        run_id: str,
        launch_token: str,
        heartbeat_at: str,
    ) -> bool:
        return heartbeat_run_worker(
            self,
            run_id=run_id,
            launch_token=launch_token,
            heartbeat_at=heartbeat_at,
        )

    def clear_run_worker(self, run_id: str, *, launch_token: str | None = None) -> None:
        clear_run_worker(self, run_id, launch_token=launch_token)

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
        upsert_run_runtime(
            self,
            run_id=run_id,
            total_timesteps=total_timesteps,
            num_timesteps=num_timesteps,
            progress_fraction=progress_fraction,
            updated_at=updated_at,
            fps=fps,
            episode_reward_mean=episode_reward_mean,
            episode_length_mean=episode_length_mean,
            approx_kl=approx_kl,
            entropy_loss=entropy_loss,
            value_loss=value_loss,
            policy_gradient_loss=policy_gradient_loss,
        )

    def append_run_event(
        self,
        *,
        run_id: str,
        kind: str,
        message: str,
        created_at: str | None = None,
    ) -> None:
        append_run_event(
            self,
            run_id=run_id,
            kind=kind,
            message=message,
            created_at=created_at,
        )

    def update_run_fork_source(
        self,
        *,
        run_id: str,
        source_snapshot_dir: Path,
        source_num_timesteps: int,
        lineage_step_offset: int | None = None,
    ) -> ManagedRun | None:
        return update_run_fork_source(
            self,
            run_id=run_id,
            source_snapshot_dir=source_snapshot_dir,
            source_num_timesteps=source_num_timesteps,
            lineage_step_offset=lineage_step_offset,
        )

    def request_run_command(
        self,
        *,
        run_id: str,
        command: RunCommand,
    ) -> ManagedRun | None:
        return request_run_command(self, run_id=run_id, command=command)

    def pending_run_command(self, run_id: str) -> RunCommand | None:
        return pending_run_command(self, run_id)

    def clear_run_command(
        self,
        run_id: str,
        *,
        command: RunCommand | None = None,
    ) -> None:
        clear_run_command(self, run_id, command=command)

    def list_drafts(self) -> tuple[ManagedRunDraft, ...]:
        return list_drafts(self)

    def delete_draft(self, draft_id: str) -> bool:
        return delete_draft(self, draft_id)

    def delete_run(self, run_id: str) -> bool:
        return delete_run(self, run_id)

    def delete_lineage(self, lineage_id: str) -> bool:
        return delete_lineage(self, lineage_id)

    def migrate_lineage_layout(self) -> int:
        return migrate_lineage_layout(self)

    def update_run_name(self, *, run_id: str, name: str) -> ManagedRun | None:
        normalized_name = name.strip() or run_id
        renamed_at = self.utc_now()
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
                    rows.run_select_sql(where_clause="WHERE runs.id = ?"),
                    (run_id,),
                ).fetchone()
        except sqlite3.IntegrityError as error:
            common.raise_name_conflict(error, table="runs", kind="run", name=normalized_name)
            raise
        return None if selected_row is None else rows.run_from_row(selected_row)

    def update_draft(
        self,
        *,
        draft_id: str,
        name: str,
        config: ManagedRunConfig,
        source_run_id: str | None = None,
        source_artifact: Literal["latest", "best"] | None = None,
    ) -> ManagedRunDraft | None:
        return update_draft(
            self,
            draft_id=draft_id,
            name=name,
            config=config,
            source_run_id=source_run_id,
            source_artifact=source_artifact,
        )

    def get_draft(self, draft_id: str) -> ManagedRunDraft | None:
        return get_draft(self, draft_id)

    def list_templates(self) -> tuple[ManagedRunTemplate, ...]:
        return list_templates(self)

    def default_template(self) -> ManagedRunTemplate:
        return default_template(self)

    def reconcile_orphaned_runs(self) -> None:
        reconcile_orphaned_runs(self)

    def _drain_pending_filesystem_operations(self) -> None:
        drain_pending_filesystem_operations(self)

    @staticmethod
    def utc_now() -> str:
        from rl_fzerox.core.manager.registry.common import utc_now

        return utc_now()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA busy_timeout = 30000")
        connection.execute("PRAGMA foreign_keys = ON")
        return connection
