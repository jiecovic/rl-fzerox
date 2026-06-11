# src/rl_fzerox/apps/viewer_runtime/lease.py
"""Process-exit cleanup for manager-owned visible viewer leases."""

from __future__ import annotations

import os
import signal
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import FrameType

from rl_fzerox.core.manager import ManagerStore
from rl_fzerox.core.manager.registry.common import utc_now
from rl_fzerox.core.manager.registry.viewers import VIEWER_LEASE_POLICY


@dataclass
class ViewerLeaseSession:
    """Runtime handle for a manager-owned visible viewer lease."""

    db_path: Path | None
    lease_id: str | None
    pid: int
    interval_seconds: float = VIEWER_LEASE_POLICY.heartbeat_interval.total_seconds()
    _last_heartbeat_monotonic: float = 0.0

    def heartbeat(self, force: bool = False) -> bool:
        if self.db_path is None or self.lease_id is None:
            return True
        now = time.monotonic()
        if not force and now - self._last_heartbeat_monotonic < self.interval_seconds:
            return True
        self._last_heartbeat_monotonic = now
        return ManagerStore(self.db_path).heartbeat_viewer_lease(
            lease_id=self.lease_id,
            pid=self.pid,
            heartbeat_at=utc_now(),
        )


@contextmanager
def manager_viewer_lease_session(
    *,
    db_path: Path | None,
    lease_id: str | None,
) -> Iterator[ViewerLeaseSession]:
    """Clear a manager viewer lease when the current process exits."""

    lease_session = ViewerLeaseSession(db_path=db_path, lease_id=lease_id, pid=os.getpid())
    cleanup = _viewer_lease_cleanup(db_path=db_path, lease_id=lease_id)
    previous_sigint = signal.getsignal(signal.SIGINT)
    previous_sigterm = signal.getsignal(signal.SIGTERM)

    def _handle_exit(signum: int, frame: FrameType | None) -> None:
        del frame
        cleanup()
        raise SystemExit(128 + signum)

    if lease_id is not None and db_path is not None:
        signal.signal(signal.SIGINT, _handle_exit)
        signal.signal(signal.SIGTERM, _handle_exit)
    try:
        lease_session.heartbeat(force=True)
        yield lease_session
    finally:
        if lease_id is not None and db_path is not None:
            signal.signal(signal.SIGINT, previous_sigint)
            signal.signal(signal.SIGTERM, previous_sigterm)
        cleanup()


def _viewer_lease_cleanup(*, db_path: Path | None, lease_id: str | None):
    process_pid = os.getpid()

    def cleanup() -> None:
        if db_path is None or lease_id is None:
            return
        try:
            ManagerStore(db_path).clear_viewer_lease(lease_id=lease_id, pid=process_pid)
        except Exception:
            # Exit cleanup must not mask the viewer's original shutdown reason.
            return

    return cleanup
