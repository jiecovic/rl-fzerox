# src/rl_fzerox/ui/watch/view/panels/text.py
from __future__ import annotations

from rl_fzerox.ui.watch.view.screen.theme import PALETTE
from rl_fzerox.ui.watch.view.screen.types import RenderFont, TextSurface


def _font_line_height(font: RenderFont) -> int:
    """Use a stable row height so glyphs like 'g' do not shift the panel."""

    return font.render("Ag", True, PALETTE.text_primary).get_height()


def _centered_text_y(y: int, row_height: int, surface: TextSurface) -> int:
    return y + max(0, (row_height - surface.get_height()) // 2)


def _fit_text(font: RenderFont, text: str, max_width: int) -> str:
    if font.render(text, True, PALETTE.text_primary).get_width() <= max_width:
        return text

    suffix = "..."
    suffix_width = font.render(suffix, True, PALETTE.text_primary).get_width()
    if suffix_width >= max_width:
        return ""

    for end_index in range(len(text), 0, -1):
        candidate = text[:end_index] + suffix
        if font.render(candidate, True, PALETTE.text_primary).get_width() <= max_width:
            return candidate
    return suffix
