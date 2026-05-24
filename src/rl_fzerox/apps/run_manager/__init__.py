# src/rl_fzerox/apps/run_manager/__init__.py
"""Run-manager package facade.

Keep the CLI entrypoint lazy so backend submodule imports do not pull the
uvicorn-backed desktop launcher into API, worker, or test paths.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rl_fzerox.apps.run_manager.app import main

__all__ = ["main"]


def __getattr__(name: str) -> object:
    if name == "main":
        from rl_fzerox.apps.run_manager.app import main

        return main
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
