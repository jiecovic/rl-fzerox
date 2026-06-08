# src/rl_fzerox/ui/watch/runtime/ipc/queues.py
from __future__ import annotations

import multiprocessing as mp
import os
from dataclasses import dataclass
from multiprocessing.queues import Queue as ProcessQueue
from queue import Empty, Full
from typing import Protocol

from fzerox_emulator import RaceControlState, SpinRequest
from rl_fzerox.core.runtime_spec.schema import WatchAppConfig
from rl_fzerox.ui.watch.input import ViewerInput
from rl_fzerox.ui.watch.runtime.cnn import (
    DEFAULT_CNN_ACTIVATION_NORMALIZATION,
    CnnActivationNormalizationMode,
)
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


def start_career_mode_worker(config: WatchAppConfig) -> WatchWorker:
    context = _multiprocessing_context()
    command_queue = context.Queue()
    snapshot_queue = context.Queue(maxsize=2)

    from rl_fzerox.ui.watch.runtime.career_mode import run_career_mode_worker

    process = context.Process(
        target=run_career_mode_worker,
        args=(config, command_queue, snapshot_queue),
        name="fzerox-career-mode",
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
    control_state: RaceControlState,
    spin_request: SpinRequest = "none",
    manual_control_enabled: bool = False,
    cnn_visualization_enabled: bool = False,
    auxiliary_visualization_enabled: bool = False,
    live_visualization_enabled: bool = False,
    cnn_normalization: CnnActivationNormalizationMode = DEFAULT_CNN_ACTIVATION_NORMALIZATION,
) -> tuple[WorkerCommandBatch, bool, RaceControlState]:
    next_paused = paused
    next_control_state = control_state
    next_spin_request = spin_request
    next_manual_control_enabled = manual_control_enabled
    quit_requested = False
    step_requests = 0
    save_requests = 0
    reset_mode: str | None = None
    jump_course_id: str | None = None
    difficulty_delta = 0
    toggle_deterministic_policy = False
    toggle_current_course_lock = False
    toggle_zeroed_state_feature_name: str | None = None
    control_fps_delta = 0
    reset_control_fps = False
    next_cnn_visualization_enabled = cnn_visualization_enabled
    next_auxiliary_visualization_enabled = auxiliary_visualization_enabled
    next_live_visualization_enabled = live_visualization_enabled
    next_cnn_normalization = cnn_normalization
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
                    reset_mode=reset_mode,
                    jump_course_id=jump_course_id,
                    difficulty_delta=difficulty_delta,
                    toggle_deterministic_policy=toggle_deterministic_policy,
                    manual_control_enabled=next_manual_control_enabled,
                    toggle_current_course_lock=toggle_current_course_lock,
                    toggle_zeroed_state_feature_name=toggle_zeroed_state_feature_name,
                    control_fps_delta=control_fps_delta,
                    reset_control_fps=reset_control_fps,
                    cnn_visualization_enabled=next_cnn_visualization_enabled,
                    auxiliary_visualization_enabled=next_auxiliary_visualization_enabled,
                    live_visualization_enabled=next_live_visualization_enabled,
                    cnn_normalization=next_cnn_normalization,
                    spin_request=next_spin_request,
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
        if command.reset_mode is not None:
            reset_mode = command.reset_mode
        if command.jump_course_id is not None:
            jump_course_id = command.jump_course_id
        difficulty_delta += command.difficulty_delta
        if command.toggle_deterministic_policy:
            toggle_deterministic_policy = not toggle_deterministic_policy
        if command.toggle_manual_control:
            next_manual_control_enabled = not next_manual_control_enabled
        if command.toggle_current_course_lock:
            toggle_current_course_lock = True
        if command.toggle_zeroed_state_feature_name is not None:
            toggle_zeroed_state_feature_name = command.toggle_zeroed_state_feature_name
        control_fps_delta += command.control_fps_delta
        reset_control_fps = reset_control_fps or command.reset_control_fps
        next_cnn_visualization_enabled = command.cnn_visualization_enabled
        next_auxiliary_visualization_enabled = command.auxiliary_visualization_enabled
        next_live_visualization_enabled = command.live_visualization_enabled
        next_cnn_normalization = command.cnn_normalization
        next_spin_request = command.spin_request
        if command.control_state is not None:
            next_control_state = command.control_state


def apply_viewer_input(
    command_queue: ProcessQueue,
    viewer_input: ViewerInput,
    *,
    paused: bool,
    cnn_visualization_enabled: bool = False,
    auxiliary_visualization_enabled: bool = False,
    live_visualization_enabled: bool = False,
    cnn_normalization: CnnActivationNormalizationMode = DEFAULT_CNN_ACTIVATION_NORMALIZATION,
) -> bool:
    next_paused = not paused if viewer_input.toggle_pause else paused
    send_command(
        command_queue,
        ViewerCommand(
            quit_requested=viewer_input.quit_requested,
            toggle_pause=viewer_input.toggle_pause,
            step_once=viewer_input.step_once,
            save_state=viewer_input.save_state,
            reset_mode=viewer_input.reset_mode,
            jump_course_id=viewer_input.jump_course_id,
            difficulty_delta=viewer_input.difficulty_delta,
            toggle_deterministic_policy=viewer_input.toggle_deterministic_policy,
            toggle_manual_control=viewer_input.toggle_manual_control,
            toggle_current_course_lock=viewer_input.toggle_current_course_lock,
            toggle_zeroed_state_feature_name=viewer_input.toggle_zeroed_state_feature_name,
            control_fps_delta=viewer_input.control_fps_delta,
            reset_control_fps=viewer_input.reset_control_fps,
            cnn_visualization_enabled=cnn_visualization_enabled,
            auxiliary_visualization_enabled=auxiliary_visualization_enabled,
            live_visualization_enabled=live_visualization_enabled,
            cnn_normalization=cnn_normalization,
            spin_request=viewer_input.spin_request,
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
