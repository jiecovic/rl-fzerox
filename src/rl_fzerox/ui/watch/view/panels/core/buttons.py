# src/rl_fzerox/ui/watch/view/panels/core/buttons.py
from __future__ import annotations

from fzerox_emulator import (
    JOYPAD_A,
    JOYPAD_B,
    JOYPAD_DOWN,
    JOYPAD_LEFT,
    JOYPAD_RIGHT,
    JOYPAD_SELECT,
    JOYPAD_START,
    JOYPAD_UP,
)

BUTTON_LABELS: tuple[tuple[int, str], ...] = (
    (JOYPAD_UP, "Up"),
    (JOYPAD_DOWN, "Down"),
    (JOYPAD_LEFT, "Left"),
    (JOYPAD_RIGHT, "Right"),
    (JOYPAD_A, "A"),
    (JOYPAD_B, "B"),
    (JOYPAD_START, "Start"),
    (JOYPAD_SELECT, "Select"),
)
