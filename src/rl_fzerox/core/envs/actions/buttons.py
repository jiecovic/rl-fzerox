# src/rl_fzerox/core/envs/actions/buttons.py
"""Semantic F-Zero X gameplay masks for policy/UI control state.

The native backend owns the concrete button bitmasks. This module re-exports
the project-facing catalog so action adapters and UI code can talk about
gameplay controls through one backend-backed vocabulary.
"""

from __future__ import annotations

from fzerox_emulator import RACE_CONTROL_MASKS, RaceControlMaskCatalog

__all__ = ["RACE_CONTROL_MASKS", "RaceControlMaskCatalog"]
