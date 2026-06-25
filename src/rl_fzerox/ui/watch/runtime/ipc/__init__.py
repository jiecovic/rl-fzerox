# src/rl_fzerox/ui/watch/runtime/ipc/__init__.py
"""Watch worker IPC facade.

This package exposes the process-boundary message types, queues, and worker
startup helpers shared by live Watch and Career Mode Watch.
"""

from __future__ import annotations

from rl_fzerox.ui.watch.runtime.ipc.messages import (
    PolicyObservationSnapshot,
    ViewerCommand,
    WatchSnapshot,
    WorkerClosed,
    WorkerCommandBatch,
    WorkerError,
)
from rl_fzerox.ui.watch.runtime.ipc.queues import (
    WatchWorker,
    WorkerMessageQueue,
    apply_viewer_input,
    drain_snapshot_queue,
    drain_worker_commands,
    publish_worker_message,
    send_command,
    start_career_mode_worker,
    start_watch_worker,
    wait_initial_snapshot,
)

__all__ = [
    "ViewerCommand",
    "PolicyObservationSnapshot",
    "WatchSnapshot",
    "WatchWorker",
    "WorkerClosed",
    "WorkerCommandBatch",
    "WorkerError",
    "WorkerMessageQueue",
    "apply_viewer_input",
    "drain_snapshot_queue",
    "drain_worker_commands",
    "publish_worker_message",
    "send_command",
    "start_career_mode_worker",
    "start_watch_worker",
    "wait_initial_snapshot",
]
