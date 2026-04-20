# src/rl_fzerox/ui/watch/view/components/cockpit/axis_geometry.py
from __future__ import annotations

from rl_fzerox.ui.watch.view.components.cockpit.style import (
    STEER_AXIS_GUIDE,
    STEER_GAUGE_STYLE,
)
from rl_fzerox.ui.watch.view.screen.theme import PALETTE, Color
from rl_fzerox.ui.watch.view.screen.types import PygameRect


def steer_indicator_color(value: float) -> Color:
    magnitude = abs(max(-1.0, min(1.0, value)))
    if magnitude <= STEER_AXIS_GUIDE.deadzone:
        return STEER_GAUGE_STYLE.deadzone_color
    if magnitude >= STEER_AXIS_GUIDE.saturation:
        return STEER_GAUGE_STYLE.extreme_color
    return PALETTE.text_primary


def steer_track_x(*, track: PygameRect, value: float) -> int:
    value = max(-1.0, min(1.0, value))
    return track.centerx + round((track.width // 2) * value)


def pitch_track_y(*, track: PygameRect, value: float) -> int:
    value = max(-1.0, min(1.0, value))
    return track.centery - round((track.height // 2) * value)


def visible_deadzone_span(
    *,
    center: int,
    lower: int,
    upper: int,
    minimum: int,
    min_bound: int,
    max_bound: int,
) -> tuple[int, int]:
    if upper - lower >= minimum:
        return lower, upper

    half_width = minimum // 2
    lower = max(min_bound, center - half_width)
    upper = min(max_bound, center + half_width)
    return lower, upper
