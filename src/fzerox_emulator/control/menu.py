# src/fzerox_emulator/control/menu.py
"""F-Zero X menu controls exposed as semantic controller masks."""

from __future__ import annotations

from dataclasses import dataclass

import fzerox_emulator._native as _native


@dataclass(frozen=True, slots=True)
class MenuButtonMasks:
    """Joypad masks for F-Zero X menu intent, not raw RetroPad button names."""

    confirm: int
    cancel: int
    start: int
    up: int
    down: int
    left: int
    right: int


def _menu_button_mask(name: str) -> int:
    value = _native.fzerox_menu_button_mask(name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must resolve to an int button mask")
    return value


MENU_BUTTON_MASKS = MenuButtonMasks(
    confirm=_menu_button_mask("confirm"),
    cancel=_menu_button_mask("cancel"),
    start=_menu_button_mask("start"),
    up=_menu_button_mask("up"),
    down=_menu_button_mask("down"),
    left=_menu_button_mask("left"),
    right=_menu_button_mask("right"),
)
