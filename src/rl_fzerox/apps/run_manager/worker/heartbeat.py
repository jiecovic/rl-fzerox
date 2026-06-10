# src/rl_fzerox/apps/run_manager/worker/heartbeat.py
from __future__ import annotations

import logging
import threading

from rl_fzerox.apps.run_manager.training_monitor import RunControlSignal
from rl_fzerox.apps.run_manager.worker.clock import now_iso
from rl_fzerox.core.manager import ManagerStore
from rl_fzerox.core.manager.registry.runs.maintenance import RUN_WORKER_LEASE_POLICY

LOGGER = logging.getLogger(__name__)


def _startup_reporter(*, store: ManagerStore, run_id: str, launch_token: str):
    def report(kind: str, message: str) -> None:
        _heartbeat_or_die(store=store, run_id=run_id, launch_token=launch_token)
        pending_command = store.pending_run_command(run_id)
        if pending_command is not None:
            raise RunControlSignal(pending_command)
        store.append_run_event(
            run_id=run_id,
            kind=kind,
            message=message,
        )

    return report


def _heartbeat_or_die(*, store: ManagerStore, run_id: str, launch_token: str) -> None:
    heartbeat_ok = store.heartbeat_run_worker(
        run_id=run_id,
        launch_token=launch_token,
        heartbeat_at=now_iso(),
    )
    if not heartbeat_ok:
        raise RuntimeError("manager worker lease is missing or stale")


class _WorkerHeartbeatLoop:
    def __init__(
        self,
        *,
        store: ManagerStore,
        run_id: str,
        launch_token: str,
        interval_seconds: float | None = None,
    ) -> None:
        self._store = store
        self._run_id = run_id
        self._launch_token = launch_token
        self._interval_seconds = (
            RUN_WORKER_LEASE_POLICY.heartbeat_interval.total_seconds()
            if interval_seconds is None
            else interval_seconds
        )
        self._stop_event = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name=f"manager-heartbeat-{run_id}",
            daemon=True,
        )

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._thread.join(timeout=self._interval_seconds + 1.0)

    def _run(self) -> None:
        while not self._stop_event.wait(self._interval_seconds):
            heartbeat_ok = self._store.heartbeat_run_worker(
                run_id=self._run_id,
                launch_token=self._launch_token,
                heartbeat_at=now_iso(),
            )
            if heartbeat_ok:
                continue
            LOGGER.error(
                "manager worker lease lost in background heartbeat run_id=%s",
                self._run_id,
            )
            return
