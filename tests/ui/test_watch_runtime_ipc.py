# tests/ui/test_watch_runtime_ipc.py
from __future__ import annotations

from queue import Empty

from fzerox_emulator import ControllerState
from rl_fzerox.ui.watch.runtime.ipc import (
    ViewerCommand,
    drain_worker_commands,
)


class _CommandQueue:
    def __init__(self, commands: list[object]) -> None:
        self._commands = commands

    def get_nowait(self) -> object:
        if not self._commands:
            raise Empty
        return self._commands.pop(0)


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
