# src/rl_fzerox/core/domain/action_adapters.py
"""Shared action-adapter name type for the managed runtime surface."""

from __future__ import annotations

from typing import Literal, TypeAlias

ActionAdapterName: TypeAlias = Literal[
    "configured_discrete",
    "configured_hybrid",
]

__all__ = [
    "ActionAdapterName",
]
