# src/fzerox_emulator/__init__.py
"""Lazy public facade for emulator exports.

Importing this package should not load the compiled native extension until one
of the exported runtime names is actually requested.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import fzerox_emulator._native as _native
    import fzerox_emulator.base as _base
    import fzerox_emulator.buttons as _buttons
    import fzerox_emulator.control as _control
    import fzerox_emulator.emulator as _emulator
    import fzerox_emulator.video as _video

    BackendStepResult = _base.BackendStepResult
    ControllerState = _control.ControllerState
    CoreInfo = _native.CoreInfo
    Emulator = _emulator.Emulator
    EmulatorBackend = _base.EmulatorBackend
    FZeroXTelemetry = _native.FZeroXTelemetry
    FrameStep = _base.FrameStep
    JOYPAD_BUTTONS = _buttons.JOYPAD_BUTTONS
    JoypadButtons = _buttons.JoypadButtons
    ObservationResizeFilter = _base.ObservationResizeFilter
    ObservationSpec = _base.ObservationSpec
    ObservationStackMode = _base.ObservationStackMode
    PlayerTelemetry = _native.PlayerTelemetry
    ResetState = _base.ResetState
    StepStatus = _native.StepStatus
    StepSummary = _native.StepSummary
    display_size = _video.display_size
    encode_state_flags = _native.encode_state_flags
    joypad_mask = _native.joypad_mask
    probe_core = _native.probe_core
    stacked_observation_channels = _base.stacked_observation_channels

_EXPORT_MODULES = {
    "CoreInfo": "fzerox_emulator._native",
    "FZeroXTelemetry": "fzerox_emulator._native",
    "PlayerTelemetry": "fzerox_emulator._native",
    "StepStatus": "fzerox_emulator._native",
    "StepSummary": "fzerox_emulator._native",
    "encode_state_flags": "fzerox_emulator._native",
    "joypad_mask": "fzerox_emulator._native",
    "probe_core": "fzerox_emulator._native",
    "BackendStepResult": "fzerox_emulator.base",
    "EmulatorBackend": "fzerox_emulator.base",
    "FrameStep": "fzerox_emulator.base",
    "ObservationResizeFilter": "fzerox_emulator.base",
    "ObservationSpec": "fzerox_emulator.base",
    "ObservationStackMode": "fzerox_emulator.base",
    "ResetState": "fzerox_emulator.base",
    "stacked_observation_channels": "fzerox_emulator.base",
    "JOYPAD_BUTTONS": "fzerox_emulator.buttons",
    "JoypadButtons": "fzerox_emulator.buttons",
    "ControllerState": "fzerox_emulator.control",
    "Emulator": "fzerox_emulator.emulator",
    "display_size": "fzerox_emulator.video",
}

__all__ = [
    "BackendStepResult",
    "ControllerState",
    "CoreInfo",
    "Emulator",
    "EmulatorBackend",
    "FZeroXTelemetry",
    "FrameStep",
    "JOYPAD_BUTTONS",
    "JoypadButtons",
    "ObservationResizeFilter",
    "ObservationSpec",
    "ObservationStackMode",
    "PlayerTelemetry",
    "ResetState",
    "StepStatus",
    "StepSummary",
    "display_size",
    "encode_state_flags",
    "joypad_mask",
    "probe_core",
    "stacked_observation_channels",
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
