# src/rl_fzerox/core/manager/db/repositories/runs/records.py
"""Repository operations for managed run identity rows."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from rl_fzerox.core.manager.db.models.runs import RunModel
from rl_fzerox.core.manager.db.models.runtime import RunWorkerModel
from rl_fzerox.core.manager.db.repositories.runs.events import append_run_event
from rl_fzerox.core.manager.db.repositories.runs.mapping import (
    _required_lineage_id,
    managed_run_from_model,
    managed_run_summary_from_model,
)
from rl_fzerox.core.manager.models import ManagedRun, ManagedRunSummary, RunStatus


def resolve_lineage_id(
    session: Session,
    *,
    explicit_lineage_id: str | None,
    parent_run_id: str | None,
    source_run_id: str | None,
    fallback_run_id: str,
) -> str:
    """Resolve the lineage id for a new run from its parent/source run."""

    if explicit_lineage_id is not None:
        return explicit_lineage_id
    parent_id = parent_run_id or source_run_id
    if parent_id is None:
        return fallback_run_id
    parent_run = session.get(RunModel, parent_id)
    if parent_run is None:
        return fallback_run_id
    return _required_lineage_id(parent_run)


def insert_run(
    session: Session,
    *,
    run: ManagedRun,
    config_snapshot_id: str,
) -> None:
    """Insert one launched run identity row."""

    session.add(
        RunModel(
            id=run.id,
            name=run.name,
            status=run.status,
            config_snapshot_id=config_snapshot_id,
            run_dir=str(run.run_dir),
            lineage_id=run.lineage_id,
            lineage_step_offset=run.lineage_step_offset,
            parent_run_id=run.parent_run_id,
            source_run_id=run.source_run_id,
            source_artifact=run.source_artifact,
            source_snapshot_dir=(
                None if run.source_snapshot_dir is None else str(run.source_snapshot_dir)
            ),
            source_num_timesteps=run.source_num_timesteps,
            created_at=run.created_at,
            started_at=run.started_at,
            stopped_at=run.stopped_at,
        )
    )


def get_managed_run(session: Session, run_id: str) -> ManagedRun | None:
    """Return one managed run by id."""

    run = session.get(RunModel, run_id)
    return None if run is None else managed_run_from_model(session, run)


def get_managed_run_summary(session: Session, run_id: str) -> ManagedRunSummary | None:
    """Return one lightweight run summary by id."""

    run = session.get(RunModel, run_id)
    return None if run is None else managed_run_summary_from_model(session, run)


def list_managed_runs(session: Session) -> tuple[ManagedRun, ...]:
    """Return all managed runs in manager display order."""

    runs = tuple(
        session.scalars(select(RunModel).order_by(RunModel.created_at.desc(), RunModel.id.desc()))
    )
    return tuple(managed_run_from_model(session, run) for run in runs)


def list_visible_managed_run_summaries(session: Session) -> tuple[ManagedRunSummary, ...]:
    """Return run-list summaries for launched runs."""

    runs = tuple(
        session.scalars(
            select(RunModel)
            .where(RunModel.status.not_in(("created", "archived")))
            .order_by(RunModel.created_at.desc(), RunModel.id.desc())
        )
    )
    return tuple(managed_run_summary_from_model(session, run) for run in runs)


def set_run_status(
    session: Session,
    *,
    run_id: str,
    status: RunStatus,
    message: str,
    started_at: str | None,
    stopped_at: str | None,
    event_at: str,
) -> ManagedRun | None:
    """Update one run status and append the corresponding event."""

    run = session.get(RunModel, run_id)
    if run is None:
        return None
    run.status = status
    if started_at is not None:
        run.started_at = started_at
    run.stopped_at = stopped_at
    if status != "running":
        worker = session.get(RunWorkerModel, run_id)
        if worker is not None:
            session.delete(worker)
    append_run_event(session, run_id=run_id, created_at=event_at, kind=status, message=message)
    session.flush()
    return managed_run_from_model(session, run)


def rename_run(
    session: Session,
    *,
    run_id: str,
    name: str,
    renamed_at: str,
) -> ManagedRun | None:
    """Rename one run and append a manager event."""

    run = session.get(RunModel, run_id)
    if run is None:
        return None
    run.name = name
    append_run_event(
        session,
        run_id=run_id,
        created_at=renamed_at,
        kind="renamed",
        message=f"run renamed to {name}",
    )
    session.flush()
    return managed_run_from_model(session, run)
