# src/fzerox_emulator/control/__init__.py
"""Controller-state and button constants for the emulator boundary."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

from fzerox_emulator.control.state import ControllerState

if TYPE_CHECKING:
    from fzerox_emulator.control.buttons import JOYPAD_BUTTONS, JoypadButtons
    from fzerox_emulator.control.spin import SpinRequest, spin_request_from_index

_EXPORT_MODULES = {
    "JOYPAD_BUTTONS": "fzerox_emulator.control.buttons",
    "JoypadButtons": "fzerox_emulator.control.buttons",
    "SpinRequest": "fzerox_emulator.control.spin",
    "spin_request_from_index": "fzerox_emulator.control.spin",
}

__all__ = [
    "ControllerState",
    "JOYPAD_BUTTONS",
    "JoypadButtons",
    "SpinRequest",
    "spin_request_from_index",
]


def __getattr__(name: str) -> object:
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(__all__)
