# src/rl_fzerox/__init__.py
"""Top-level package facade.

Keep the environment export lazy so plain package imports do not pull the
emulator-bound env implementation into unrelated tooling paths.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rl_fzerox.core.envs import FZeroXEnv

__all__ = ["FZeroXEnv"]


def __getattr__(name: str) -> object:
    if name == "FZeroXEnv":
        from rl_fzerox.core.envs import FZeroXEnv

        return FZeroXEnv
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
