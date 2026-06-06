# src/rl_fzerox/apps/viewer_runtime/lease.py
"""Process-exit cleanup for manager-owned visible viewer leases."""

from __future__ import annotations

import os
import signal
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import FrameType

from rl_fzerox.core.manager import ManagerStore


@contextmanager
def manager_viewer_lease_session(
    *,
    db_path: Path | None,
    lease_id: str | None,
) -> Iterator[None]:
    """Clear a manager viewer lease when the current process exits."""

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
        yield
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
