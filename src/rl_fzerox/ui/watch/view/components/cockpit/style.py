# src/rl_fzerox/ui/watch/view/components/cockpit/style.py
from __future__ import annotations

from dataclasses import dataclass, field

from rl_fzerox.core.envs.state_observation import DEFAULT_STATE_VECTOR_SPEC
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
class PolicyModeSwitchStyle:
    """Top-down switch sprite for deterministic versus stochastic playback."""

    switch_width: int = 50
    height: int = 42
    label_gap: int = 8
    y_offset: int = 2
    asset_package: str = "rl_fzerox.ui.watch.view.assets"
    on_asset: str = "policy_switch_on.png"
    off_asset: str = "policy_switch_off.png"
    disabled_asset: str = "policy_switch_disabled.png"
    active_text: Color = (244, 218, 132)
    inactive_text: Color = PALETTE.text_muted


@dataclass(frozen=True)
class CockpitPanelStyle:
    """Shared cockpit panel colors."""

    wide_control_min_width: int = 420
    wide_panel_height: int = 204
    compact_panel_height: int = 144
    fill: Color = (14, 19, 25)
    border: Color = (74, 91, 112)
    grid: Color = (31, 42, 54)
    policy_mode_switch: PolicyModeSwitchStyle = field(default_factory=PolicyModeSwitchStyle)


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
    deadzone_marker: Color = (96, 165, 250)
    full_zone_marker: Color = PALETTE.text_accent
    unlit_segment_fill: Color = CockpitPanelStyle().grid
    border_width: int = 2
    marker_width: int = 2
    marker_extension: int = 5
    tall_segment_count: int = 9
    compact_segment_count: int = 7
    tall_segment_gap: int = 2
    compact_segment_gap: int = 1
    tall_segment_height: int = 9
    tall_segment_trimmed_height: int = 8
    compact_segment_height: int = 8
    segment_horizontal_inset: int = 4


@dataclass(frozen=True)
class SpeedGaugeStyle:
    """Compact analog speedometer used inside the cockpit panel."""

    render_scale: int = 3
    max_kph: float = DEFAULT_STATE_VECTOR_SPEC.speed_normalizer_kph
    red_zone_start_kph: float = 800.0
    start_degrees: float = 200.0
    sweep_degrees: float = 220.0
    tick_count: int = 5
    arc_segments: int = 44
    arc_width: int = 2
    active_arc_width: int = 3
    energy_meter_width: int = 10
    energy_meter_height: int = 54
    energy_meter_margin_right: int = 9
    energy_meter_center_y_offset: int = -2
    energy_meter_padding: int = 2
    energy_meter_radius: int = 4
    energy_outer_border_width: int = 2
    energy_inner_border_width: int = 1
    radius_padding: int = 9
    vertical_radius_padding: int = 42
    center_bottom_padding: int = 32
    needle_width: int = 2
    needle_length_fraction: float = 0.78
    label_y_offset: int = 20
    fill: Color = CockpitPanelStyle().fill
    inner_fill: Color = ButtonFaceStyle().inner
    border: Color = CockpitPanelStyle().border
    inner_border: Color = PALETTE.panel_border
    scanline: Color = CockpitPanelStyle().grid
    arc_idle: Color = PALETTE.control_track
    red_zone: Color = (68, 48, 38)
    tick: Color = PALETTE.control_lever_border
    needle: Color = PALETTE.text_warning
    needle_shadow: Color = (3, 5, 8)
    pivot: Color = ButtonFaceStyle().inner
    active_low: Color = PALETTE.text_accent
    active_high: Color = PALETTE.text_warning
    energy_track: Color = CockpitPanelStyle().grid
    energy_border: Color = (82, 62, 118)
    energy_inner_border: Color = (236, 220, 255)
    energy_empty_fill: Color = (18, 24, 31)
    energy_fill: Color = (176, 128, 255)
    energy_low_fill: Color = (224, 96, 76)
    energy_fill_lip: Color = (226, 206, 255)
    energy_glass_highlight: AlphaColor = (245, 235, 255, 58)
    energy_glass_shadow: AlphaColor = (0, 0, 0, 52)
    energy_glass_edge: AlphaColor = (210, 236, 224, 42)

    @property
    def red_zone_start(self) -> float:
        """Normalized speed where the warning arc begins."""

        return max(0.0, min(1.0, self.red_zone_start_kph / self.max_kph))


@dataclass(frozen=True)
class AvailabilityLedStyle:
    """Small status lamps showing whether a control branch is currently usable."""

    radius: int = 4
    inner_radius: int = 2
    glow_radius: int = 10
    on_fill: Color = (228, 112, 28)
    on_inner: Color = (255, 210, 94)
    on_border: Color = (255, 184, 80)
    on_glow: AlphaColor = (255, 126, 26, 92)
    off_fill: Color = (28, 32, 36)
    off_inner: Color = (49, 55, 61)
    off_border: Color = (94, 86, 74)
    shadow: Color = (4, 6, 8)
    highlight: AlphaColor = (255, 239, 184, 92)


STEER_AXIS_GUIDE = SteerAxisGuide()
LEAN_CONTROL_STYLE = LeanControlStyle()
STEER_GAUGE_STYLE = SteerGaugeStyle()
COCKPIT_PANEL_STYLE = CockpitPanelStyle()
BUTTON_FACE_STYLE = ButtonFaceStyle()
THRUST_COLUMN_STYLE = ThrustColumnStyle()
SPEED_GAUGE_STYLE = SpeedGaugeStyle()
AVAILABILITY_LED_STYLE = AvailabilityLedStyle()
