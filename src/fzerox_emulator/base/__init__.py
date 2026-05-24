# src/fzerox_emulator/base/__init__.py
from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fzerox_emulator.base.backend import EmulatorBackend
    from fzerox_emulator.base.observations import (
        FrameObservationOptions,
        ObservationImageRecipe,
        ObservationResizeFilter,
        ObservationSpec,
        ObservationStackMode,
        stacked_observation_channels,
    )
    from fzerox_emulator.base.results import (
        BackendMultiObservationStepResult,
        BackendStepResult,
        FrameStep,
        ResetState,
    )

_EXPORT_MODULES = {
    "EmulatorBackend": "fzerox_emulator.base.backend",
    "FrameObservationOptions": "fzerox_emulator.base.observations",
    "ObservationImageRecipe": "fzerox_emulator.base.observations",
    "ObservationResizeFilter": "fzerox_emulator.base.observations",
    "ObservationSpec": "fzerox_emulator.base.observations",
    "ObservationStackMode": "fzerox_emulator.base.observations",
    "stacked_observation_channels": "fzerox_emulator.base.observations",
    "BackendMultiObservationStepResult": "fzerox_emulator.base.results",
    "BackendStepResult": "fzerox_emulator.base.results",
    "FrameStep": "fzerox_emulator.base.results",
    "ResetState": "fzerox_emulator.base.results",
}

__all__ = [
    "BackendMultiObservationStepResult",
    "BackendStepResult",
    "EmulatorBackend",
    "FrameObservationOptions",
    "FrameStep",
    "ObservationImageRecipe",
    "ObservationResizeFilter",
    "ObservationSpec",
    "ObservationStackMode",
    "ResetState",
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
