# src/rl_fzerox/core/manager/db/repositories/viewers.py
"""Repository operations for manager-owned visible viewer leases."""

from __future__ import annotations

from sqlalchemy.orm import Session

from rl_fzerox.core.manager.db.models.runtime import ViewerLeaseModel
from rl_fzerox.core.manager.models import ManagedViewerLease, ViewerLeaseKind


def upsert_viewer_lease(session: Session, lease: ManagedViewerLease) -> None:
    """Create or replace one visible viewer process lease."""

    row = session.get(ViewerLeaseModel, lease.id)
    if row is None:
        session.add(
            ViewerLeaseModel(
                id=lease.id,
                kind=lease.kind,
                owner_id=lease.owner_id,
                qualifier=lease.qualifier,
                pid=lease.pid,
                launched_at=lease.launched_at,
                heartbeat_at=lease.heartbeat_at,
            )
        )
        return

    row.kind = lease.kind
    row.owner_id = lease.owner_id
    row.qualifier = lease.qualifier
    row.pid = lease.pid
    row.launched_at = lease.launched_at
    row.heartbeat_at = lease.heartbeat_at


def get_viewer_lease(session: Session, lease_id: str) -> ManagedViewerLease | None:
    """Return one viewer process lease by id."""

    row = session.get(ViewerLeaseModel, lease_id)
    return None if row is None else viewer_lease_from_model(row)


def clear_viewer_lease(
    session: Session,
    *,
    lease_id: str,
    pid: int | None = None,
) -> bool:
    """Delete one viewer lease, optionally only when it still belongs to a pid."""

    row = session.get(ViewerLeaseModel, lease_id)
    if row is None or (pid is not None and row.pid != pid):
        return False
    session.delete(row)
    return True


def viewer_lease_from_model(row: ViewerLeaseModel) -> ManagedViewerLease:
    """Convert one ORM row into a domain viewer lease."""

    return ManagedViewerLease(
        id=row.id,
        kind=_viewer_lease_kind(row.kind),
        owner_id=row.owner_id,
        qualifier=row.qualifier,
        pid=row.pid,
        launched_at=row.launched_at,
        heartbeat_at=row.heartbeat_at,
    )


def _viewer_lease_kind(value: object) -> ViewerLeaseKind:
    match value:
        case "run_watch":
            return "run_watch"
        case "career_mode":
            return "career_mode"
    raise ValueError(f"Unsupported viewer lease kind: {value!r}")
