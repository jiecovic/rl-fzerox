# src/fzerox_emulator/control/buttons.py
from __future__ import annotations

from dataclasses import dataclass

import fzerox_emulator._native as _native


@dataclass(frozen=True, slots=True)
class JoypadButtons:
    b: int
    y: int
    select: int
    start: int
    up: int
    down: int
    left: int
    right: int
    a: int
    x: int
    left_shoulder: int
    right_shoulder: int
    left_trigger: int
    right_trigger: int
    left_stick: int
    right_stick: int


def _button(name: str) -> int:
    value = getattr(_native, name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int button id")
    return value


JOYPAD_BUTTONS = JoypadButtons(
    b=_button("JOYPAD_B"),
    y=_button("JOYPAD_Y"),
    select=_button("JOYPAD_SELECT"),
    start=_button("JOYPAD_START"),
    up=_button("JOYPAD_UP"),
    down=_button("JOYPAD_DOWN"),
    left=_button("JOYPAD_LEFT"),
    right=_button("JOYPAD_RIGHT"),
    a=_button("JOYPAD_A"),
    x=_button("JOYPAD_X"),
    left_shoulder=_button("JOYPAD_L"),
    right_shoulder=_button("JOYPAD_R"),
    left_trigger=_button("JOYPAD_L2"),
    right_trigger=_button("JOYPAD_R2"),
    left_stick=_button("JOYPAD_L3"),
    right_stick=_button("JOYPAD_R3"),
)
