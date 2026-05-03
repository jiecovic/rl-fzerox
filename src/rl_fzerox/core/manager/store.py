# src/rl_fzerox/core/manager/store.py
from __future__ import annotations

import re
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from rl_fzerox.core.manager.config import ManagedRunConfig
from rl_fzerox.core.manager.errors import ManagerNameConflictError
from rl_fzerox.core.manager.models import (
    ManagedRun,
    ManagedRunDraft,
    ManagedRunTemplate,
    RunStatus,
)
from rl_fzerox.core.manager.schema import initialize_manager_schema
from rl_fzerox.core.manager.serialization import config_hash, config_json, load_config_json


def default_manager_db_path() -> Path:
    """Return the default local manager registry path."""

    return Path("local/manager/runs.db").resolve()


class ManagerStore:
    """SQLite-backed source of truth for managed training runs."""

    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = (db_path or default_manager_db_path()).expanduser().resolve()

    def initialize(self) -> None:
        """Create the manager database schema if needed."""

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            initialize_manager_schema(connection, applied_at=_utc_now())

    def create_run(
        self,
        *,
        name: str,
        config: ManagedRunConfig,
        managed_runs_root: Path | None = None,
        parent_run_id: str | None = None,
        source_run_id: str | None = None,
        source_artifact: str | None = None,
    ) -> ManagedRun:
        """Create an immutable SQLite run record without filesystem side effects."""

        self.initialize()
        created_at = _utc_now()
        run_id = _new_run_id(name)
        normalized_name = name.strip() or run_id
        root = (managed_runs_root or Path("local/managed_runs")).expanduser().resolve()
        run_dir = root / run_id
        run = ManagedRun(
            id=run_id,
            name=normalized_name,
            status="created",
            config=config,
            config_hash=config_hash(config),
            run_dir=run_dir,
            parent_run_id=parent_run_id,
            source_run_id=source_run_id,
            source_artifact=source_artifact,
            created_at=created_at,
        )

        try:
            with self._connect() as connection:
                _assert_name_available(connection, normalized_name)
                connection.execute(
                    """
                    INSERT INTO runs(
                        id,
                        name,
                        status,
                        config_json,
                        config_hash,
                        run_dir,
                        parent_run_id,
                        source_run_id,
                        source_artifact,
                        created_at,
                        started_at,
                        stopped_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        run.id,
                        run.name,
                        run.status,
                        config_json(run.config),
                        run.config_hash,
                        str(run.run_dir),
                        run.parent_run_id,
                        run.source_run_id,
                        run.source_artifact,
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
        return run

    def create_draft(
        self,
        *,
        name: str,
        config: ManagedRunConfig,
    ) -> ManagedRunDraft:
        """Persist a mutable draft in SQLite without creating filesystem artifacts."""

        self.initialize()
        created_at = _utc_now()
        draft_id = _new_record_id(name)
        normalized_name = name.strip() or draft_id
        draft = ManagedRunDraft(
            id=draft_id,
            name=normalized_name,
            config=config,
            config_hash=config_hash(config),
            created_at=created_at,
            updated_at=created_at,
        )

        try:
            with self._connect() as connection:
                _assert_name_available(connection, normalized_name)
                connection.execute(
                    """
                    INSERT INTO run_drafts(
                        id,
                        name,
                        config_json,
                        config_hash,
                        created_at,
                        updated_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        draft.id,
                        draft.name,
                        config_json(draft.config),
                        draft.config_hash,
                        draft.created_at,
                        draft.updated_at,
                    ),
                )
        except sqlite3.IntegrityError as error:
            _raise_name_conflict(error, table="run_drafts", kind="draft", name=normalized_name)
            raise
        return draft

    def list_runs(self) -> tuple[ManagedRun, ...]:
        """Return all DB-managed runs, newest first."""

        self.initialize()
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    id,
                    name,
                    status,
                    config_json,
                    config_hash,
                    run_dir,
                    parent_run_id,
                    source_run_id,
                    source_artifact,
                    created_at,
                    started_at,
                    stopped_at
                FROM runs
                ORDER BY created_at DESC, id DESC
                """
            ).fetchall()
        return tuple(_run_from_row(row) for row in rows)

    def list_visible_runs(self) -> tuple[ManagedRun, ...]:
        """Return runs that should appear in the current run registry UI."""

        return tuple(run for run in self.list_runs() if run.status != "created")

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
                    created_at,
                    updated_at
                FROM run_drafts
                ORDER BY updated_at DESC, id DESC
                """
            ).fetchall()
        return tuple(_draft_from_row(row) for row in rows)

    def delete_draft(self, draft_id: str) -> bool:
        """Delete one SQLite-only draft by id."""

        self.initialize()
        with self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM run_drafts WHERE id = ?",
                (draft_id,),
            )
        return cursor.rowcount > 0

    def update_draft(
        self,
        *,
        draft_id: str,
        name: str,
        config: ManagedRunConfig,
    ) -> ManagedRunDraft | None:
        """Update one SQLite-backed draft in place."""

        self.initialize()
        updated_at = _utc_now()
        normalized_name = name.strip() or draft_id
        try:
            with self._connect() as connection:
                _assert_name_available(connection, normalized_name, exclude_draft_id=draft_id)
                row = connection.execute(
                    """
                    UPDATE run_drafts
                    SET
                        name = ?,
                        config_json = ?,
                        config_hash = ?,
                        updated_at = ?
                    WHERE id = ?
                    RETURNING
                        id,
                        name,
                        config_json,
                        config_hash,
                        created_at,
                        updated_at
                    """,
                    (
                        normalized_name,
                        config_json(config),
                        config_hash(config),
                        updated_at,
                        draft_id,
                    ),
                ).fetchone()
        except sqlite3.IntegrityError as error:
            _raise_name_conflict(error, table="run_drafts", kind="draft", name=normalized_name)
            raise
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

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
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
        parent_run_id=_optional_str(row["parent_run_id"]),
        source_run_id=_optional_str(row["source_run_id"]),
        source_artifact=_optional_str(row["source_artifact"]),
        created_at=str(row["created_at"]),
        started_at=_optional_str(row["started_at"]),
        stopped_at=_optional_str(row["stopped_at"]),
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


def _optional_str(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _new_run_id(name: str) -> str:
    return _new_record_id(name)


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
    exclude_run_id: str | None = None,
) -> None:
    row = connection.execute(
        """
        SELECT source FROM (
            SELECT 'run' AS source
            FROM runs
            WHERE name = ? COLLATE NOCASE
              AND (? IS NULL OR id != ?)
            UNION ALL
            SELECT 'draft' AS source
            FROM run_drafts
            WHERE name = ? COLLATE NOCASE
              AND (? IS NULL OR id != ?)
        )
        LIMIT 1
        """,
        (
            name,
            exclude_run_id,
            exclude_run_id,
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
