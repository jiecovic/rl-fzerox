# src/rl_fzerox/ui/watch/runtime/ipc/queues.py
from __future__ import annotations

import multiprocessing as mp
import os
from dataclasses import dataclass
from multiprocessing.queues import Queue as ProcessQueue
from queue import Empty, Full
from typing import Protocol

from fzerox_emulator import ControllerState
from rl_fzerox.core.config.schema import WatchAppConfig
from rl_fzerox.ui.watch.input import ViewerInput
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


class WatchWorkerProcess(Protocol):
    def join(self, timeout: float | None = None) -> None: ...

    def is_alive(self) -> bool: ...

    def terminate(self) -> None: ...

    def kill(self) -> None: ...


@dataclass
class WatchWorker:
    """Process handle plus queues used by the watch UI."""

    process: WatchWorkerProcess
    command_queue: ProcessQueue
    snapshot_queue: ProcessQueue

    def shutdown(self) -> None:
        try:
            send_command(self.command_queue, ViewerCommand(quit_requested=True))
        except (BrokenPipeError, EOFError, OSError, ValueError):
            pass

        interrupted = _safe_join(self.process, timeout=0.25)
        if self.process.is_alive():
            interrupted = _force_stop_process(self.process) or interrupted

        _close_queue(self.command_queue)
        _close_queue(self.snapshot_queue)
        if interrupted:
            return


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


def _safe_join(process: WatchWorkerProcess, *, timeout: float) -> bool:
    try:
        process.join(timeout=timeout)
    except KeyboardInterrupt:
        return True
    return False


def _force_stop_process(process: WatchWorkerProcess) -> bool:
    interrupted = False
    process.terminate()
    interrupted = _safe_join(process, timeout=0.5) or interrupted
    if process.is_alive():
        process.kill()
        interrupted = _safe_join(process, timeout=0.25) or interrupted
    return interrupted


def send_command(command_queue: ProcessQueue, command: ViewerCommand) -> None:
    command_queue.put(command)


def drain_worker_commands(
    command_queue: _ReadableCommandQueue,
    *,
    paused: bool,
    control_state: ControllerState,
    manual_control_enabled: bool = False,
    cnn_visualization_enabled: bool = False,
) -> tuple[WorkerCommandBatch, bool, ControllerState]:
    next_paused = paused
    next_control_state = control_state
    next_manual_control_enabled = manual_control_enabled
    quit_requested = False
    step_requests = 0
    save_requests = 0
    reset_requested = False
    toggle_deterministic_policy = False
    control_fps_delta = 0
    next_cnn_visualization_enabled = cnn_visualization_enabled
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
                    toggle_deterministic_policy=toggle_deterministic_policy,
                    manual_control_enabled=next_manual_control_enabled,
                    control_fps_delta=control_fps_delta,
                    cnn_visualization_enabled=next_cnn_visualization_enabled,
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
        if command.toggle_deterministic_policy:
            toggle_deterministic_policy = not toggle_deterministic_policy
        if command.toggle_manual_control:
            next_manual_control_enabled = not next_manual_control_enabled
        control_fps_delta += command.control_fps_delta
        next_cnn_visualization_enabled = command.cnn_visualization_enabled
        if command.control_state is not None:
            next_control_state = command.control_state


def apply_viewer_input(
    command_queue: ProcessQueue,
    viewer_input: ViewerInput,
    *,
    paused: bool,
    cnn_visualization_enabled: bool = False,
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
            toggle_deterministic_policy=viewer_input.toggle_deterministic_policy,
            toggle_manual_control=viewer_input.toggle_manual_control,
            control_fps_delta=viewer_input.control_fps_delta,
            cnn_visualization_enabled=cnn_visualization_enabled,
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
