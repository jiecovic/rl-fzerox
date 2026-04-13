# src/rl_fzerox/ui/watch/layout.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

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

Color = tuple[int, int, int]


class TextSurface(Protocol):
    """Minimal surface contract used by the viewer layout helpers."""

    def get_width(self) -> int: ...

    def get_height(self) -> int: ...


class RenderFont(Protocol):
    """Minimal font contract used by the viewer renderer."""

    def render(
        self,
        text: str,
        antialias: bool,
        color: Color,
    ) -> TextSurface: ...


@dataclass(frozen=True)
class ViewerLayout:
    """Spacing and sizing used by the watch window layout."""

    panel_width: int = 600
    panel_min_height: int = 800
    panel_padding: int = 12
    preview_gap: int = 12
    preview_scale: int = 1
    preview_padding: int = 12
    preview_title_gap: int = 6
    column_gap: int = 16
    title_gap: int = 2
    title_section_gap: int = 8
    section_gap: int = 8
    section_title_gap: int = 4
    section_rule_gap: int = 4
    line_gap: int = 2
    inline_value_gap: int = 8
    wrapped_value_indent: int = 10
    control_viz_gap: int = 3
    control_widget_gap: int = 12
    control_track_gap: int = 4
    control_side_pill_gap: int = 8
    control_steer_width: int = 116
    control_steer_height: int = 10
    control_drive_width: int = 12
    control_drive_height: int = 56
    control_drive_pair_gap: int = 48
    control_drive_offset_x: int = 8
    control_marker_radius: int = 6
    control_caption_gap: int = 3
    control_boost_gap: int = 4
    flag_token_gap: int = 4
    flag_token_pad_x: int = 6
    flag_token_pad_y: int = 2


@dataclass(frozen=True)
class ViewerPalette:
    """Color palette for the watch window."""

    app_background: Color = (11, 14, 18)
    panel_background: Color = (20, 24, 30)
    panel_border: Color = (46, 54, 66)
    text_primary: Color = (238, 241, 245)
    text_muted: Color = (155, 165, 178)
    text_accent: Color = (126, 214, 170)
    text_warning: Color = (241, 206, 108)
    control_track: Color = (39, 46, 56)
    control_knob: Color = (238, 241, 245)
    control_knob_outline: Color = (20, 24, 30)
    control_coast: Color = (86, 97, 112)
    flag_active_background: Color = (36, 68, 56)
    flag_active_border: Color = (126, 214, 170)
    flag_inactive_background: Color = (27, 33, 41)
    flag_inactive_border: Color = (46, 54, 66)


@dataclass(frozen=True)
class PanelLine:
    """One labeled line in the side panel."""

    label: str
    value: str
    color: Color
    wrap: bool = False
    min_value_lines: int = 1


@dataclass(frozen=True)
class ControlViz:
    """Compact control-state visualization for the watch panel."""

    steer_x: float
    drive_level: int
    drive_axis: float | None
    air_brake_axis: float | None
    air_brake_disabled: bool
    drive_axis_mode: Literal["signed", "accelerate"]
    boost_pressed: bool
    drift_direction: int


@dataclass(frozen=True)
class FlagToken:
    """One compact flag label rendered inside the watch panel."""

    label: str
    active: bool


@dataclass(frozen=True)
class FlagViz:
    """Fixed multi-row racer flag visualization for the watch panel."""

    rows: tuple[tuple[FlagToken, ...], ...]


@dataclass(frozen=True)
class PanelSection:
    """One titled section in the side panel."""

    title: str
    lines: list[PanelLine]
    control_viz: ControlViz | None = None
    flag_viz: FlagViz | None = None


@dataclass(frozen=True)
class PanelColumns:
    """Left and right panel columns rendered next to the game view."""

    left: list[PanelSection]
    right: list[PanelSection]


@dataclass(frozen=True)
class ViewerFonts:
    """Font bundle used by the watch panel renderer."""

    title: RenderFont
    section: RenderFont
    body: RenderFont
    small: RenderFont


@dataclass(frozen=True)
class ViewerFontSizes:
    """Point sizes for the watch panel fonts."""

    title: int = 24
    section: int = 20
    # Mono fonts have taller metrics than pygame's default font; 12 matches the
    # old body text height closely while keeping value columns tabular.
    body: int = 12
    small: int = 16


LAYOUT = ViewerLayout()
PALETTE = ViewerPalette()
FONT_SIZES = ViewerFontSizes()

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
