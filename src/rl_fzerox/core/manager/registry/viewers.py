# src/rl_fzerox/core/manager/registry/viewers.py
"""Manager-store operations for visible viewer process leases."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rl_fzerox.core.manager.db.repositories import viewers as viewer_repository
from rl_fzerox.core.manager.models import ManagedViewerLease, ViewerLeaseKind
from rl_fzerox.core.manager.registry.common import utc_now

if TYPE_CHECKING:
    from rl_fzerox.core.manager.store import ManagerStore


def viewer_lease_id(
    *,
    kind: ViewerLeaseKind,
    owner_id: str,
    qualifier: str | None = None,
) -> str:
    """Return the stable manager lease id for one visible viewer process."""

    suffix = "" if qualifier is None else f":{qualifier}"
    return f"{kind}:{owner_id}{suffix}"


def upsert_viewer_lease(
    store: ManagerStore,
    *,
    lease_id: str,
    kind: ViewerLeaseKind,
    owner_id: str,
    pid: int,
    qualifier: str | None = None,
) -> ManagedViewerLease:
    """Create or replace the live process lease for a viewer."""

    store.initialize()
    now = utc_now()
    lease = ManagedViewerLease(
        id=lease_id,
        kind=kind,
        owner_id=owner_id,
        qualifier=qualifier,
        pid=pid,
        launched_at=now,
        heartbeat_at=now,
    )
    with store._orm_session() as session:
        viewer_repository.upsert_viewer_lease(session, lease)
    return lease


def get_viewer_lease(store: ManagerStore, lease_id: str) -> ManagedViewerLease | None:
    """Return one visible viewer process lease, if still registered."""

    store.initialize()
    with store._orm_session() as session:
        return viewer_repository.get_viewer_lease(session, lease_id)


def clear_viewer_lease(
    store: ManagerStore,
    *,
    lease_id: str,
    pid: int | None = None,
) -> bool:
    """Clear one viewer process lease, optionally only for the owning process."""

    store.initialize()
    with store._orm_session() as session:
        return viewer_repository.clear_viewer_lease(session, lease_id=lease_id, pid=pid)
