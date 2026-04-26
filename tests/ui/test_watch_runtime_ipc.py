# tests/ui/test_watch_runtime_ipc.py
from __future__ import annotations

from queue import Empty

from pytest import MonkeyPatch

from fzerox_emulator import ControllerState
from rl_fzerox.ui.watch.runtime.ipc import (
    ViewerCommand,
    WatchWorker,
    WorkerClosed,
    drain_worker_commands,
)
from rl_fzerox.ui.watch.runtime.worker import run_simulation_worker


class _CommandQueue:
    def __init__(self, commands: list[object]) -> None:
        self._commands = commands

    def get_nowait(self) -> object:
        if not self._commands:
            raise Empty
        return self._commands.pop(0)


class _ShutdownQueue:
    def __init__(self) -> None:
        self.items: list[object] = []
        self.cancelled = False
        self.closed = False

    def put(self, obj: object) -> None:
        self.items.append(obj)

    def cancel_join_thread(self) -> None:
        self.cancelled = True

    def close(self) -> None:
        self.closed = True


class _InterruptingProcess:
    def __init__(self) -> None:
        self._alive = True
        self.join_calls: list[float | None] = []
        self.terminate_calls = 0
        self.kill_calls = 0

    def join(self, timeout: float | None = None) -> None:
        self.join_calls.append(timeout)
        if len(self.join_calls) == 1:
            raise KeyboardInterrupt

    def is_alive(self) -> bool:
        return self._alive

    def terminate(self) -> None:
        self.terminate_calls += 1
        self._alive = False

    def kill(self) -> None:
        self.kill_calls += 1
        self._alive = False


def test_drain_worker_commands_coalesces_force_reset() -> None:
    command_queue = _CommandQueue([ViewerCommand(force_reset=True)])

    commands, paused, control_state = drain_worker_commands(
        command_queue,
        paused=False,
        control_state=ControllerState(),
    )

    assert commands.reset_requested is True
    assert paused is False
    assert control_state == ControllerState()


def test_drain_worker_commands_coalesces_deterministic_toggle_parity() -> None:
    command_queue = _CommandQueue(
        [
            ViewerCommand(toggle_deterministic_policy=True),
            ViewerCommand(toggle_deterministic_policy=True),
            ViewerCommand(toggle_deterministic_policy=True),
        ]
    )

    commands, paused, control_state = drain_worker_commands(
        command_queue,
        paused=False,
        control_state=ControllerState(),
    )

    assert commands.toggle_deterministic_policy is True
    assert paused is False
    assert control_state == ControllerState()


def test_drain_worker_commands_preserves_cnn_visualization_state_without_commands() -> None:
    command_queue = _CommandQueue([])

    commands, paused, control_state = drain_worker_commands(
        command_queue,
        paused=False,
        control_state=ControllerState(),
        cnn_visualization_enabled=True,
    )

    assert commands.cnn_visualization_enabled is True
    assert paused is False
    assert control_state == ControllerState()


def test_drain_worker_commands_updates_cnn_visualization_state() -> None:
    command_queue = _CommandQueue([ViewerCommand(cnn_visualization_enabled=False)])

    commands, _, _ = drain_worker_commands(
        command_queue,
        paused=False,
        control_state=ControllerState(),
        cnn_visualization_enabled=True,
    )

    assert commands.cnn_visualization_enabled is False


def test_drain_worker_commands_updates_cnn_normalization() -> None:
    command_queue = _CommandQueue([ViewerCommand(cnn_normalization="layer_percentile")])

    commands, _, _ = drain_worker_commands(
        command_queue,
        paused=False,
        control_state=ControllerState(),
        cnn_normalization="channel",
    )

    assert commands.cnn_normalization == "layer_percentile"


def test_drain_worker_commands_toggles_manual_control_state() -> None:
    command_queue = _CommandQueue([ViewerCommand(toggle_manual_control=True)])

    commands, _, _ = drain_worker_commands(
        command_queue,
        paused=False,
        control_state=ControllerState(),
        manual_control_enabled=False,
    )

    assert commands.manual_control_enabled is True


def test_drain_worker_commands_coalesces_course_lock_toggle() -> None:
    command_queue = _CommandQueue(
        [
            ViewerCommand(toggle_track_course_lock_id="mute_city"),
            ViewerCommand(toggle_track_course_lock_id="silence"),
        ]
    )

    commands, _, _ = drain_worker_commands(
        command_queue,
        paused=False,
        control_state=ControllerState(),
    )

    assert commands.toggle_track_course_lock_id == "silence"


def test_watch_worker_shutdown_swallows_keyboard_interrupt_during_join() -> None:
    command_queue = _ShutdownQueue()
    snapshot_queue = _ShutdownQueue()
    process = _InterruptingProcess()
    worker = WatchWorker(
        process=process,
        command_queue=command_queue,  # type: ignore[arg-type]  # queue test double
        snapshot_queue=snapshot_queue,  # type: ignore[arg-type]  # queue test double
    )

    worker.shutdown()

    assert len(command_queue.items) == 1
    command = command_queue.items[0]
    assert isinstance(command, ViewerCommand)
    assert command.quit_requested is True
    assert process.terminate_calls == 1
    assert command_queue.cancelled is True
    assert command_queue.closed is True
    assert snapshot_queue.cancelled is True
    assert snapshot_queue.closed is True


def test_run_simulation_worker_swallows_keyboard_interrupt(
    monkeypatch: MonkeyPatch,
) -> None:
    published: list[object] = []

    def _raise_keyboard_interrupt(*_args: object, **_kwargs: object) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(
        "rl_fzerox.ui.watch.runtime.worker._run_simulation_loop",
        _raise_keyboard_interrupt,
    )
    monkeypatch.setattr(
        "rl_fzerox.ui.watch.runtime.worker.publish_worker_message",
        lambda _queue, message: published.append(message),
    )

    run_simulation_worker(
        config=object(),  # type: ignore[arg-type]
        command_queue=object(),  # type: ignore[arg-type]
        snapshot_queue=object(),  # type: ignore[arg-type]
    )

    assert len(published) == 1
    assert isinstance(published[0], WorkerClosed)
