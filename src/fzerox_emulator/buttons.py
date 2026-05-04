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
    l: int
    r: int
    l2: int
    r2: int
    l3: int
    r3: int


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
    l=_button("JOYPAD_L"),
    r=_button("JOYPAD_R"),
    l2=_button("JOYPAD_L2"),
    r2=_button("JOYPAD_R2"),
    l3=_button("JOYPAD_L3"),
    r3=_button("JOYPAD_R3"),
)
