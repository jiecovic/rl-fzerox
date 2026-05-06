# src/rl_fzerox/core/manager/registry/drafts/store.py
from __future__ import annotations

import sqlite3
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from rl_fzerox.core.manager.models import ManagedRunDraft, ManagedRunTemplate
from rl_fzerox.core.manager.registry.common import (
    assert_name_available,
    new_record_id,
    raise_name_conflict,
    utc_now,
)
from rl_fzerox.core.manager.registry.drafts.fork_sources import (
    reset_draft_source,
    snapshot_draft_source,
)
from rl_fzerox.core.manager.registry.rows import draft_from_row, template_from_row
from rl_fzerox.core.manager.run_spec import ManagedRunConfig
from rl_fzerox.core.manager.storage.serialization import config_hash, config_json

if TYPE_CHECKING:
    from rl_fzerox.core.manager.store import ManagerStore


def create_draft(
    store: ManagerStore,
    *,
    name: str,
    config: ManagedRunConfig,
    source_run_id: str | None = None,
    source_artifact: Literal["latest", "best"] | None = None,
) -> ManagedRunDraft:
    """Persist one mutable draft and pin a fork source when requested."""

    store.initialize()
    created_at = utc_now()
    draft_id = new_record_id(name)
    normalized_name = name.strip() or draft_id
    source_snapshot_dir = None
    source_num_timesteps = None
    source_run = None if source_run_id is None else store.get_run(source_run_id)
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
        with store._connect() as connection:
            assert_name_available(connection, normalized_name)
            if source_run_id is not None and source_artifact is not None:
                assert source_run is not None
                source_snapshot_dir, source_num_timesteps = snapshot_draft_source(
                    manager_db_path=store.db_path,
                    draft_id=draft_id,
                    source_run=source_run,
                    source_artifact=source_artifact,
                )
                draft = replace(
                    draft,
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
                    None if draft.source_snapshot_dir is None else str(draft.source_snapshot_dir),
                    draft.source_num_timesteps,
                    draft.created_at,
                    draft.updated_at,
                ),
            )
    except sqlite3.IntegrityError as error:
        raise_name_conflict(error, table="run_drafts", kind="draft", name=normalized_name)
        raise
    except Exception:
        reset_draft_source(source_snapshot_dir)
        raise
    return draft


def list_drafts(store: ManagerStore) -> tuple[ManagedRunDraft, ...]:
    store.initialize()
    with store._connect() as connection:
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
    return tuple(draft_from_row(row) for row in rows)


def delete_draft(store: ManagerStore, draft_id: str) -> bool:
    store.initialize()
    with store._connect() as connection:
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
        reset_draft_source(Path(str(row["source_snapshot_dir"])).expanduser().resolve())
    return cursor.rowcount > 0


def update_draft(
    store: ManagerStore,
    *,
    draft_id: str,
    name: str,
    config: ManagedRunConfig,
    source_run_id: str | None = None,
    source_artifact: Literal["latest", "best"] | None = None,
) -> ManagedRunDraft | None:
    store.initialize()
    updated_at = utc_now()
    normalized_name = name.strip() or draft_id
    current_draft = store.get_draft(draft_id)
    if current_draft is None:
        return None
    source_run = None if source_run_id is None else store.get_run(source_run_id)
    if source_run_id is not None and source_run is None:
        raise ValueError(f"run not found: {source_run_id}")
    next_snapshot_dir = current_draft.source_snapshot_dir
    next_source_num_timesteps = current_draft.source_num_timesteps
    source_changed = (
        current_draft.source_run_id != source_run_id
        or current_draft.source_artifact != source_artifact
    )
    if source_changed and current_draft.source_run_id is not None:
        raise ValueError("changing a fork draft source is not supported; create a new fork draft")
    try:
        with store._connect() as connection:
            assert_name_available(connection, normalized_name, exclude_draft_id=draft_id)
            if source_changed:
                next_snapshot_dir = None
                next_source_num_timesteps = None
                if source_run_id is not None and source_artifact is not None:
                    assert source_run is not None
                    next_snapshot_dir, next_source_num_timesteps = snapshot_draft_source(
                        manager_db_path=store.db_path,
                        draft_id=draft_id,
                        source_run=source_run,
                        source_artifact=source_artifact,
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
        raise_name_conflict(error, table="run_drafts", kind="draft", name=normalized_name)
        raise
    return None if row is None else draft_from_row(row)


def get_draft(store: ManagerStore, draft_id: str) -> ManagedRunDraft | None:
    store.initialize()
    with store._connect() as connection:
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
    return None if row is None else draft_from_row(row)


def list_templates(store: ManagerStore) -> tuple[ManagedRunTemplate, ...]:
    store.initialize()
    with store._connect() as connection:
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
    return tuple(template_from_row(row) for row in rows)


def default_template(store: ManagerStore) -> ManagedRunTemplate:
    templates = list_templates(store)
    if not templates:
        raise RuntimeError("Manager DB did not initialize a default run template")
    return templates[0]
