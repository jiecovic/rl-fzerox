# src/fzerox_emulator/control/state.py
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ControllerState:
    """Normalized controller state used by the emulator boundary.

    Analog axes are expressed in the normalized range ``[-1.0, 1.0]``. The
    native host clamps them before converting to libretro's signed 16-bit range.
    """

    joypad_mask: int = 0
    left_stick_x: float = 0.0
    left_stick_y: float = 0.0
    right_stick_x: float = 0.0
    right_stick_y: float = 0.0
