# src/fzerox_emulator/repeat/__init__.py
"""Lazy facade for native repeated-step helper functions."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

from fzerox_emulator.repeat.step_options import RepeatStepConfig

if TYPE_CHECKING:
    from fzerox_emulator.repeat.multi import run_repeat_multi_observation_step
    from fzerox_emulator.repeat.single import run_repeat_step
    from fzerox_emulator.repeat.watch import run_repeat_watch_step

__all__ = [
    "RepeatStepConfig",
    "run_repeat_multi_observation_step",
    "run_repeat_step",
    "run_repeat_watch_step",
]

_EXPORT_MODULES = {
    "run_repeat_multi_observation_step": "fzerox_emulator.repeat.multi",
    "run_repeat_step": "fzerox_emulator.repeat.single",
    "run_repeat_watch_step": "fzerox_emulator.repeat.watch",
}


def __getattr__(name: str) -> object:
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(__all__)
