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
    left_width = (content_width - LAYOUT.column_gap) // 2
    right_width = content_width - LAYOUT.column_gap - left_width
    left_height = _column_content_height(fonts, columns.left, width=left_width)
    right_height = _column_content_height(fonts, columns.right, width=right_width)
    return y + max(left_height, right_height) + LAYOUT.panel_padding
