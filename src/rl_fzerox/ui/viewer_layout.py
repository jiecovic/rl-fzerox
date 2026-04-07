# src/rl_fzerox/ui/viewer_layout.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from rl_fzerox._native import (
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

    panel_width: int = 456
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


@dataclass(frozen=True)
class PanelLine:
    """One labeled line in the side panel."""

    label: str
    value: str
    color: Color


@dataclass(frozen=True)
class PanelSection:
    """One titled section in the side panel."""

    title: str
    lines: list[PanelLine]


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
    body: int = 18
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
