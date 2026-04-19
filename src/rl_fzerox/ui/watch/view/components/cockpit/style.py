# src/rl_fzerox/ui/watch/view/components/cockpit/style.py
from __future__ import annotations

from dataclasses import dataclass

from rl_fzerox.ui.watch.view.screen.theme import PALETTE, Color

AlphaColor = tuple[int, int, int, int]
Offset2D = tuple[int, int]


@dataclass(frozen=True)
class SteerAxisGuide:
    """Approximate F-Zero X steering thresholds in policy action space."""

    deadzone: float = 0.08
    saturation: float = 63.0 / 80.0


@dataclass(frozen=True)
class LeanControlStyle:
    """Visual constants for the compact L/R half-moon lean controls."""

    height_extra: int = 10
    y_offset: int = -4
    inner_inset: int = 5
    glow_inflate: int = 4
    curve_steps: int = 18
    label_bottom_padding: int = 5
    flat_edge_inset: int = 2
    shadow_offset: Offset2D = (1, 2)
    inactive_inner_fill: Color = (18, 24, 31)
    inactive_border: Color = (75, 87, 100)
    inactive_text: Color = PALETTE.text_muted
    active_fill: Color = (27, 82, 66)
    active_border: Color = (126, 214, 170)
    active_glow_alpha: int = 38
    sheen: AlphaColor = (238, 248, 255, 82)


@dataclass(frozen=True)
class SteerGaugeStyle:
    """Visual constants for the retro steering rail gauge."""

    render_scale: int = 3
    target_inflate: Offset2D = (16, 14)
    bezel_inflate: Offset2D = (8, 6)
    rail_inset: Offset2D = (5, 3)
    shadow_offset: Offset2D = (1, 2)
    rail_vertical_shrink: int = 6
    fill_vertical_shrink: int = 2
    deadzone_visual_min_size: int = 16
    deadzone_tick_extension: int = 3
    deadzone_tick_width: int = 2
    bezel_radius: int = 4
    rail_radius: int = 2
    shadow_fill: Color = (7, 10, 14)
    bezel_fill: Color = (15, 20, 26)
    bezel_border: Color = (68, 79, 92)
    rail_fill: Color = (22, 28, 35)
    center_line: Color = (134, 149, 166)
    deadzone_line: Color = (154, 197, 249)
    deadzone_color: Color = (96, 165, 250)
    extreme_color: Color = PALETTE.text_accent
    glass_tint: AlphaColor = (176, 190, 184, 24)
    glass_highlight: AlphaColor = (232, 238, 224, 38)
    glass_shadow: AlphaColor = (0, 0, 0, 28)
    glass_edge: Color = (145, 164, 156)
    pointer_glint: AlphaColor = (255, 255, 255, 76)


@dataclass(frozen=True)
class CockpitPanelStyle:
    """Shared cockpit panel colors."""

    wide_control_min_width: int = 420
    fill: Color = (14, 19, 25)
    border: Color = (74, 91, 112)
    grid: Color = (31, 42, 54)


@dataclass(frozen=True)
class ButtonFaceStyle:
    """Shared hard-surface button colors."""

    shadow: Color = (7, 10, 14)
    fill: Color = (24, 31, 39)
    highlight: Color = (62, 76, 92)
    inner: Color = (17, 23, 30)
    active_text: Color = (232, 255, 246)


@dataclass(frozen=True)
class ThrustColumnStyle:
    """Shared vertical level-meter styling for thrust and brake columns."""

    warning_fill: Color = (241, 206, 108)
    unlit_segment_fill: Color = CockpitPanelStyle().grid
    border_width: int = 2


STEER_AXIS_GUIDE = SteerAxisGuide()
LEAN_CONTROL_STYLE = LeanControlStyle()
STEER_GAUGE_STYLE = SteerGaugeStyle()
COCKPIT_PANEL_STYLE = CockpitPanelStyle()
BUTTON_FACE_STYLE = ButtonFaceStyle()
THRUST_COLUMN_STYLE = ThrustColumnStyle()
