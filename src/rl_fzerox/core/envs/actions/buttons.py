# src/rl_fzerox/core/envs/actions/buttons.py
"""Canonical RetroPad-to-F-Zero control masks.

The emulator button wiring is fixed for the managed runtime surface, so this
module keeps one owned mapping catalog and exports stable aliases for the rest
of the env and watch code.
"""

from __future__ import annotations

from dataclasses import dataclass

from fzerox_emulator import JOYPAD_BUTTONS, joypad_mask


@dataclass(frozen=True, slots=True)
class RaceControlMaskCatalog:
    """RetroPad button masks for the fixed managed control surface."""

    # Mupen64Plus-Next's standard RetroPad mapping exposes the in-game N64
    # A/C-Down buttons on RetroPad B/A. F-Zero X names them Accelerate and
    # Air Brake on the gameplay side.
    accelerate: int = joypad_mask(JOYPAD_BUTTONS.b)
    air_brake: int = joypad_mask(JOYPAD_BUTTONS.a)

    # N64 B/Z/R map to RetroPad Y/L2/R and correspond to boost and lean.
    boost: int = joypad_mask(JOYPAD_BUTTONS.y)
    lean_left: int = joypad_mask(JOYPAD_BUTTONS.left_trigger)
    lean_right: int = joypad_mask(JOYPAD_BUTTONS.right_shoulder)


RACE_CONTROL_MASKS = RaceControlMaskCatalog()
