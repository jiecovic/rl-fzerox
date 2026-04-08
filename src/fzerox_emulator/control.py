# src/fzerox_emulator/control.py
from __future__ import annotations

from dataclasses import dataclass


def _clamp_axis(value: float) -> float:
    return max(-1.0, min(1.0, value))


@dataclass(frozen=True, slots=True)
class ControllerState:
    """Normalized libretro controller state used by the emulator boundary.

    Analog axes are expressed in the normalized range ``[-1.0, 1.0]`` and are
    converted to libretro's native signed 16-bit range inside the Rust host.
    """

    joypad_mask: int = 0
    left_stick_x: float = 0.0
    left_stick_y: float = 0.0
    right_stick_x: float = 0.0
    right_stick_y: float = 0.0

    def clamped(self) -> ControllerState:
        """Return the same controller state with all axes clamped to range."""

        return ControllerState(
            joypad_mask=self.joypad_mask,
            left_stick_x=_clamp_axis(self.left_stick_x),
            left_stick_y=_clamp_axis(self.left_stick_y),
            right_stick_x=_clamp_axis(self.right_stick_x),
            right_stick_y=_clamp_axis(self.right_stick_y),
        )
