# src/rl_fzerox/core/manager/registry/runs/lifecycle.py
"""Run creation, lookup, and lifecycle status registry operations."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from sqlalchemy.exc import IntegrityError as SqlAlchemyIntegrityError

from rl_fzerox.core.manager.db.repositories.configs import create_config_snapshot
from rl_fzerox.core.manager.db.repositories.runs import (
    append_run_event as insert_run_event,
)
from rl_fzerox.core.manager.db.repositories.runs import (
    get_managed_run,
    get_managed_run_summary,
    insert_run,
    list_managed_runs,
    list_recent_managed_run_events,
    list_visible_managed_run_summaries,
    rename_run,
    resolve_lineage_id,
    set_run_status,
)
from rl_fzerox.core.manager.models import (
    ManagedRun,
    ManagedRunEvent,
    ManagedRunSummary,
    RunStatus,
)
from rl_fzerox.core.manager.registry.common import (
    new_run_id,
    raise_name_conflict,
    utc_now,
)
from rl_fzerox.core.manager.run_spec import ManagedRunConfig

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
    from rl_fzerox.core.manager.registry.paths import manager_run_dir

    store.initialize()
    created_at = utc_now()
    resolved_run_id = run_id or new_run_id()
    normalized_name = name.strip() or resolved_run_id
    root = (managed_runs_root or store.manager_runs_root()).expanduser().resolve()
    run: ManagedRun | None = None

    try:
        with store._orm_session() as session:
            resolved_lineage_id = resolve_lineage_id(
                session,
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
            config_snapshot = create_config_snapshot(
                session,
                kind="run",
                config=config,
                created_at=created_at,
            )
            run = ManagedRun(
                id=resolved_run_id,
                name=normalized_name,
                status="created",
                config=config,
                config_hash=config_snapshot.config_hash,
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
            insert_run(
                session,
                run=run,
                config_snapshot_id=config_snapshot.id,
            )
            session.flush()
            insert_run_event(
                session,
                run_id=run.id,
                created_at=created_at,
                kind="created",
                message="run created from manager config",
            )
    except SqlAlchemyIntegrityError as error:
        raise_name_conflict(
            sqlite3.IntegrityError(str(error.orig)),
            table="runs",
            kind="run",
            name=normalized_name,
        )
        raise
    if run is None:
        raise RuntimeError("managed run creation failed before insert")
    return run


def get_run(store: ManagerStore, run_id: str) -> ManagedRun | None:
    store.initialize()
    with store._orm_session() as session:
        return get_managed_run(session, run_id)


def get_run_summary(store: ManagerStore, run_id: str) -> ManagedRunSummary | None:
    store.initialize()
    with store._orm_session() as session:
        return get_managed_run_summary(session, run_id)


def list_runs(store: ManagerStore) -> tuple[ManagedRun, ...]:
    store.initialize()
    with store._orm_session() as session:
        return list_managed_runs(session)


def list_visible_runs(store: ManagerStore) -> tuple[ManagedRun, ...]:
    return tuple(run for run in list_runs(store) if run.status not in {"created", "archived"})


def list_visible_run_summaries(store: ManagerStore) -> tuple[ManagedRunSummary, ...]:
    store.initialize()
    with store._orm_session() as session:
        return list_visible_managed_run_summaries(session)


def list_recent_run_events(
    store: ManagerStore,
    run_ids: tuple[str, ...],
    *,
    limit_per_run: int = 6,
) -> dict[str, tuple[ManagedRunEvent, ...]]:
    if not run_ids:
        return {}
    store.initialize()
    with store._orm_session() as session:
        return list_recent_managed_run_events(
            session,
            run_ids,
            limit_per_run=limit_per_run,
        )


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
    with store._orm_session() as session:
        return set_run_status(
            session,
            run_id=run_id,
            status=status,
            message=message,
            started_at=started_at,
            stopped_at=stopped_at,
            event_at=event_at,
        )


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
        with store._orm_session() as session:
            return rename_run(
                session,
                run_id=run_id,
                name=normalized_name,
                renamed_at=renamed_at,
            )
    except SqlAlchemyIntegrityError as error:
        raise_name_conflict(
            sqlite3.IntegrityError(str(error.orig)),
            table="runs",
            kind="run",
            name=normalized_name,
        )
        raise
    except sqlite3.IntegrityError as error:
        raise_name_conflict(error, table="runs", kind="run", name=normalized_name)
        raise
