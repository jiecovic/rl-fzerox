# src/rl_fzerox/ui/watch/view/panels/core/buttons.py
from __future__ import annotations

from fzerox_emulator import JOYPAD_BUTTONS

BUTTON_LABELS: tuple[tuple[int, str], ...] = (
    (JOYPAD_BUTTONS.up, "Up"),
    (JOYPAD_BUTTONS.down, "Down"),
    (JOYPAD_BUTTONS.left, "Left"),
    (JOYPAD_BUTTONS.right, "Right"),
    (JOYPAD_BUTTONS.a, "A"),
    (JOYPAD_BUTTONS.b, "B"),
    (JOYPAD_BUTTONS.start, "Start"),
    (JOYPAD_BUTTONS.select, "Select"),
)
