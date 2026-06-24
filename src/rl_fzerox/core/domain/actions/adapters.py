# src/rl_fzerox/core/domain/actions/adapters.py
"""Shared action-adapter name type for the managed runtime surface."""

from __future__ import annotations

from typing import Literal

type ActionAdapterName = Literal[
    "configured_discrete",
    "configured_hybrid",
]

__all__ = [
    "ActionAdapterName",
]
