# src/rl_fzerox/core/training/session/callbacks/sb3/__init__.py
"""SB3 callback factory facade."""

from __future__ import annotations

from .factory import build_callbacks

__all__ = ["build_callbacks"]
