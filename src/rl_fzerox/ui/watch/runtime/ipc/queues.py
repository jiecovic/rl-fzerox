# src/rl_fzerox/ui/watch/runtime/ipc/queues.py
from __future__ import annotations

import multiprocessing as mp
import os
from dataclasses import dataclass
from multiprocessing.process import BaseProcess
from multiprocessing.queues import Queue as ProcessQueue
from queue import Empty, Full
from typing import Protocol

from fzerox_emulator import ControllerState
from rl_fzerox.core.config.schema import WatchAppConfig
from rl_fzerox.ui.watch.runtime.ipc.messages import (
    ViewerCommand,
    WatchSnapshot,
    WorkerClosed,
    WorkerCommandBatch,
    WorkerError,
)


class _ReadableCommandQueue(Protocol):
    def get_nowait(self) -> object: ...


class WorkerMessageQueue(Protocol):
    def put_nowait(self, obj: object) -> None: ...

    def get_nowait(self) -> object: ...


@dataclass
class WatchWorker:
    """Process handle plus queues used by the watch UI."""

    process: BaseProcess
    command_queue: ProcessQueue
    snapshot_queue: ProcessQueue

    def shutdown(self) -> None:
        send_command(self.command_queue, ViewerCommand(quit_requested=True))
        self.process.join(timeout=0.25)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=0.5)
        _close_queue(self.command_queue)
        _close_queue(self.snapshot_queue)


def start_watch_worker(config: WatchAppConfig) -> WatchWorker:
    context = _multiprocessing_context()
    command_queue = context.Queue()
    snapshot_queue = context.Queue(maxsize=2)

    from rl_fzerox.ui.watch.runtime.worker import run_simulation_worker

    process = context.Process(
        target=run_simulation_worker,
        args=(config, command_queue, snapshot_queue),
        name="fzerox-watch-sim",
    )
    process.daemon = True
    process.start()
    return WatchWorker(
        process=process,
        command_queue=command_queue,
        snapshot_queue=snapshot_queue,
    )


def _multiprocessing_context():
    if hasattr(os, "fork"):
        return mp.get_context("fork")
    return mp.get_context("spawn")


def _close_queue(queue: ProcessQueue) -> None:
    queue.cancel_join_thread()
    queue.close()


def send_command(command_queue: ProcessQueue, command: ViewerCommand) -> None:
    command_queue.put(command)


def drain_worker_commands(
    command_queue: _ReadableCommandQueue,
    *,
    paused: bool,
    control_state: ControllerState,
) -> tuple[WorkerCommandBatch, bool, ControllerState]:
    next_paused = paused
    next_control_state = control_state
    quit_requested = False
    step_requests = 0
    save_requests = 0
    reset_requested = False
    control_fps_delta = 0
    while True:
        try:
            command = command_queue.get_nowait()
        except Empty:
            return (
                WorkerCommandBatch(
                    quit_requested=quit_requested,
                    paused=next_paused,
                    step_requests=step_requests,
                    save_requests=save_requests,
                    reset_requested=reset_requested,
                    control_fps_delta=control_fps_delta,
                    control_state=next_control_state,
                ),
                next_paused,
                next_control_state,
            )
        if not isinstance(command, ViewerCommand):
            continue
        if command.quit_requested:
            quit_requested = True
        if command.toggle_pause:
            next_paused = not next_paused
        if command.step_once:
            step_requests += 1
        if command.save_state:
            save_requests += 1
        if command.force_reset:
            reset_requested = True
        control_fps_delta += command.control_fps_delta
        if command.control_state is not None:
            next_control_state = command.control_state


def apply_viewer_input(
    command_queue: ProcessQueue,
    viewer_input,
    *,
    paused: bool,
) -> bool:
    next_paused = not paused if viewer_input.toggle_pause else paused
    send_command(
        command_queue,
        ViewerCommand(
            quit_requested=viewer_input.quit_requested,
            toggle_pause=viewer_input.toggle_pause,
            step_once=viewer_input.step_once,
            save_state=viewer_input.save_state,
            force_reset=viewer_input.force_reset,
            control_fps_delta=viewer_input.control_fps_delta,
            control_state=viewer_input.control_state,
        ),
    )
    return next_paused


def wait_initial_snapshot(worker: WatchWorker) -> tuple[WatchSnapshot, bool]:
    while True:
        try:
            message = worker.snapshot_queue.get(timeout=0.1)
        except Empty:
            if not worker.process.is_alive():
                raise RuntimeError(
                    "watch simulation worker stopped before publishing a frame"
                ) from None
            continue
        if isinstance(message, WatchSnapshot):
            return message, False
        if isinstance(message, WorkerError):
            raise RuntimeError(message.message)
        if isinstance(message, WorkerClosed):
            raise RuntimeError("watch simulation worker stopped before publishing a frame")


def drain_snapshot_queue(
    worker: WatchWorker,
    *,
    worker_closed: bool,
) -> tuple[WatchSnapshot | None, bool]:
    latest: WatchSnapshot | None = None
    closed = worker_closed
    while True:
        try:
            message = worker.snapshot_queue.get_nowait()
        except Empty:
            return latest, closed
        if isinstance(message, WatchSnapshot):
            latest = message
        elif isinstance(message, WorkerError):
            raise RuntimeError(message.message)
        elif isinstance(message, WorkerClosed):
            closed = True


def publish_worker_message(
    snapshot_queue: WorkerMessageQueue,
    message: WatchSnapshot | WorkerError | WorkerClosed,
) -> None:
    while True:
        try:
            snapshot_queue.put_nowait(message)
            return
        except Full:
            try:
                snapshot_queue.get_nowait()
            except Empty:
                pass
