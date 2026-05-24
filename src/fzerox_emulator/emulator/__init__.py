# src/fzerox_emulator/emulator/__init__.py
"""Lazy facade for the concrete Python emulator wrapper."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fzerox_emulator.emulator.client import Emulator

__all__ = ["Emulator"]

_EXPORT_MODULES = {
    "Emulator": "fzerox_emulator.emulator.client",
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
