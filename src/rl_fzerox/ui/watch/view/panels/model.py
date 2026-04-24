# src/rl_fzerox/ui/watch/view/panels/model.py
from __future__ import annotations

from rl_fzerox.ui.watch.view.panels import sections as _sections
from rl_fzerox.ui.watch.view.screen import observation_preview as _preview
from rl_fzerox.ui.watch.view.screen.layout import LAYOUT
from rl_fzerox.ui.watch.view.screen.theme import PALETTE
from rl_fzerox.ui.watch.view.screen.types import PanelColumns, ViewerFonts

_build_panel_columns = _sections._build_panel_columns
_column_content_height = _sections._column_content_height
_observation_preview_size = _preview._observation_preview_size
_preview_block_height = _preview._preview_block_height
_preview_frame = _preview._preview_frame
_preview_panel_size = _preview._preview_panel_size
_window_size = _preview._window_size


def _panel_content_height(
    fonts: ViewerFonts,
    columns: PanelColumns,
    *,
    observation_shape: tuple[int, ...],
) -> int:
    """Return the content-driven height of the watch HUD side panel."""

    title_surface = fonts.title.render("F-Zero X Watch", True, PALETTE.text_primary)
    subtitle_surface = fonts.small.render("live emulator session", True, PALETTE.text_muted)
    y = LAYOUT.panel_padding
    y += title_surface.get_height() + LAYOUT.title_gap + subtitle_surface.get_height()
    y += LAYOUT.title_section_gap

    content_width = LAYOUT.panel_width - (2 * LAYOUT.panel_padding)
    tab_width = _panel_tab_width(content_width)
    tab_height = fonts.small.render("Session", True, PALETTE.text_primary).get_height() + 10
    left_height = _column_content_height(fonts, columns.left, width=tab_width)
    middle_height = _column_content_height(fonts, columns.middle, width=tab_width)
    stats_height = _column_content_height(fonts, columns.stats, width=tab_width)
    return (
        y
        + tab_height
        + LAYOUT.title_section_gap
        + max(left_height, middle_height, stats_height)
        + LAYOUT.panel_padding
    )


def _panel_column_widths(content_width: int) -> tuple[int, int, int]:
    """Return balanced widths for the three watch side-panel columns."""

    available_width = content_width - (2 * LAYOUT.column_gap)
    base_width, remainder = divmod(available_width, 3)
    return (
        base_width,
        base_width + (1 if remainder >= 1 else 0),
        base_width + (1 if remainder >= 2 else 0),
    )


def _panel_tab_width(content_width: int) -> int:
    """Return the fixed-width selected HUD tab column."""

    return max(1, min(LAYOUT.panel_tab_width, content_width))


def _panel_preview_width(content_width: int) -> int:
    """Return the width reserved for the observation preview column."""

    tab_width = _panel_tab_width(content_width)
    return max(0, content_width - tab_width - LAYOUT.column_gap)
