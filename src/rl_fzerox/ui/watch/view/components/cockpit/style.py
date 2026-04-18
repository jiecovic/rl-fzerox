# src/rl_fzerox/ui/watch/view/components/cockpit/style.py
from __future__ import annotations

from dataclasses import dataclass

from rl_fzerox.ui.watch.view.screen.theme import PALETTE, Color


@dataclass(frozen=True)
class SteerAxisGuide:
    """Approximate F-Zero X steering thresholds in policy action space."""

    deadzone: float = 0.08
    saturation: float = 63.0 / 80.0


STEER_AXIS_GUIDE = SteerAxisGuide()
STEER_DEADZONE_COLOR: Color = (96, 165, 250)
STEER_EXTREME_COLOR: Color = PALETTE.text_accent
COCKPIT_PANEL_FILL: Color = (14, 19, 25)
COCKPIT_PANEL_BORDER: Color = (74, 91, 112)
COCKPIT_GRID: Color = (31, 42, 54)
THRUST_WARNING_FILL: Color = (241, 206, 108)
THRUST_COLUMN_BORDER_WIDTH = 2
LEAN_ACTIVE_FILL: Color = (27, 82, 66)
LEAN_ACTIVE_BORDER: Color = (126, 214, 170)
BUTTON_SHADOW: Color = (7, 10, 14)
BUTTON_FACE_FILL: Color = (24, 31, 39)
BUTTON_FACE_HIGHLIGHT: Color = (62, 76, 92)
BUTTON_FACE_INNER: Color = (17, 23, 30)
BUTTON_ACTIVE_TEXT: Color = (232, 255, 246)
