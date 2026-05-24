# src/rl_fzerox/core/envs/__init__.py
"""Lazy env facade.

Importing the env package should not instantiate the full emulator-facing env
module graph until callers actually request `FZeroXEnv`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rl_fzerox.core.envs.env import FZeroXEnv

__all__ = ["FZeroXEnv"]


def __getattr__(name: str) -> object:
    if name == "FZeroXEnv":
        from rl_fzerox.core.envs.env import FZeroXEnv

        return FZeroXEnv
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
