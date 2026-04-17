# tests/ui/test_watch_runtime_process.py
from __future__ import annotations

from queue import Empty

from fzerox_emulator import ControllerState
from rl_fzerox.ui.watch.runtime.process import (
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
