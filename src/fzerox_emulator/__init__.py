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
    import fzerox_emulator.base.backend as _backend
    import fzerox_emulator.base.observations as _observations
    import fzerox_emulator.base.results as _results
    import fzerox_emulator.control as _control
    import fzerox_emulator.emulator as _emulator

    BackendMultiObservationStepResult = _results.BackendMultiObservationStepResult
    BackendStepResult = _results.BackendStepResult
    ControllerState = _control.ControllerState
    CoreInfo = _native.CoreInfo
    Emulator = _emulator.Emulator
    EmulatorBackend = _backend.EmulatorBackend
    FZeroXTelemetry = _native.FZeroXTelemetry
    FrameStep = _results.FrameStep
    JOYPAD_BUTTONS = _control.JOYPAD_BUTTONS
    JoypadButtons = _control.JoypadButtons
    ObservationResizeFilter = _observations.ObservationResizeFilter
    ObservationImageRecipe = _observations.ObservationImageRecipe
    ObservationSpec = _observations.ObservationSpec
    ObservationStackMode = _observations.ObservationStackMode
    PlayerTelemetry = _native.PlayerTelemetry
    RACE_CONTROL_MASKS = _control.RACE_CONTROL_MASKS
    RaceControlMaskCatalog = _control.RaceControlMaskCatalog
    RaceControlState = _control.RaceControlState
    ResetState = _results.ResetState
    StepStatus = _native.StepStatus
    StepSummary = _native.StepSummary
    SpinRequest = _control.SpinRequest
    display_size = _observations.display_size
    encode_state_flags = _native.encode_state_flags
    joypad_mask = _native.joypad_mask
    probe_core = _native.probe_core
    stacked_observation_channels = _observations.stacked_observation_channels

_EXPORT_MODULES = {
    "CoreInfo": "fzerox_emulator._native",
    "FZeroXTelemetry": "fzerox_emulator._native",
    "PlayerTelemetry": "fzerox_emulator._native",
    "StepStatus": "fzerox_emulator._native",
    "StepSummary": "fzerox_emulator._native",
    "encode_state_flags": "fzerox_emulator._native",
    "joypad_mask": "fzerox_emulator._native",
    "probe_core": "fzerox_emulator._native",
    "BackendStepResult": "fzerox_emulator.base.results",
    "BackendMultiObservationStepResult": "fzerox_emulator.base.results",
    "EmulatorBackend": "fzerox_emulator.base.backend",
    "FrameStep": "fzerox_emulator.base.results",
    "ObservationResizeFilter": "fzerox_emulator.base.observations",
    "ObservationImageRecipe": "fzerox_emulator.base.observations",
    "ObservationSpec": "fzerox_emulator.base.observations",
    "ObservationStackMode": "fzerox_emulator.base.observations",
    "ResetState": "fzerox_emulator.base.results",
    "stacked_observation_channels": "fzerox_emulator.base.observations",
    "JOYPAD_BUTTONS": "fzerox_emulator.control",
    "JoypadButtons": "fzerox_emulator.control",
    "RACE_CONTROL_MASKS": "fzerox_emulator.control",
    "RaceControlMaskCatalog": "fzerox_emulator.control",
    "RaceControlState": "fzerox_emulator.control",
    "SpinRequest": "fzerox_emulator.control",
    "ControllerState": "fzerox_emulator.control",
    "Emulator": "fzerox_emulator.emulator",
    "display_size": "fzerox_emulator.base.observations",
}

__all__ = [
    "BackendMultiObservationStepResult",
    "BackendStepResult",
    "ControllerState",
    "CoreInfo",
    "Emulator",
    "EmulatorBackend",
    "FZeroXTelemetry",
    "FrameStep",
    "JOYPAD_BUTTONS",
    "JoypadButtons",
    "ObservationImageRecipe",
    "ObservationResizeFilter",
    "ObservationSpec",
    "ObservationStackMode",
    "PlayerTelemetry",
    "RACE_CONTROL_MASKS",
    "RaceControlMaskCatalog",
    "RaceControlState",
    "ResetState",
    "StepStatus",
    "StepSummary",
    "SpinRequest",
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
