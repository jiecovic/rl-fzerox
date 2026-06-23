# src/rl_fzerox/core/manager/db/repositories/runs/events.py
"""Repository operations for run event rows."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from rl_fzerox.core.manager.db.models.runs import RunEventModel
from rl_fzerox.core.manager.models import ManagedRunEvent


def append_run_event(
    session: Session,
    *,
    run_id: str,
    created_at: str,
    kind: str,
    message: str,
) -> None:
    """Append one event row for a run."""

    session.add(
        RunEventModel(
            run_id=run_id,
            created_at=created_at,
            kind=kind,
            message=message,
        )
    )


def list_recent_managed_run_events(
    session: Session,
    run_ids: tuple[str, ...],
    *,
    limit_per_run: int,
) -> dict[str, tuple[ManagedRunEvent, ...]]:
    """Return the most recent events per requested run id."""

    if not run_ids:
        return {}
    events = tuple(
        session.scalars(
            select(RunEventModel)
            .where(RunEventModel.run_id.in_(run_ids))
            .order_by(RunEventModel.created_at.desc(), RunEventModel.id.desc())
        )
    )
    events_by_run_id: dict[str, list[ManagedRunEvent]] = {run_id: [] for run_id in run_ids}
    for event in events:
        run_events = events_by_run_id.setdefault(event.run_id, [])
        if len(run_events) >= limit_per_run:
            continue
        run_events.append(
            ManagedRunEvent(
                run_id=event.run_id,
                created_at=event.created_at,
                kind=event.kind,
                message=event.message,
            )
        )
    return {
        event_run_id: tuple(run_events)
        for event_run_id, run_events in events_by_run_id.items()
        if run_events
    }
