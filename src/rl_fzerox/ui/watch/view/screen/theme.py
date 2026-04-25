# src/rl_fzerox/ui/watch/view/screen/theme.py
from __future__ import annotations

from dataclasses import dataclass

Color = tuple[int, int, int]


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
    cnn_grid_separator: Color = (54, 132, 140)
    cnn_grid_unused_hatch: Color = (47, 91, 96)
    control_track: Color = (39, 46, 56)
    control_lever_border: Color = (142, 153, 168)
    control_lever_border_width: int = 1
    control_knob: Color = (238, 241, 245)
    control_knob_outline: Color = (20, 24, 30)
    control_coast: Color = (86, 97, 112)
    flag_active_background: Color = (36, 68, 56)
    flag_active_border: Color = (126, 214, 170)
    flag_inactive_background: Color = (27, 33, 41)
    flag_inactive_border: Color = (46, 54, 66)


@dataclass(frozen=True)
class ViewerFontSizes:
    """Point sizes for the watch panel fonts."""

    title: int = 24
    section: int = 20
    # Mono fonts have taller metrics than pygame's default font; 12 matches the
    # old body text height closely while keeping value columns tabular.
    body: int = 12
    small: int = 16


PALETTE = ViewerPalette()
FONT_SIZES = ViewerFontSizes()
