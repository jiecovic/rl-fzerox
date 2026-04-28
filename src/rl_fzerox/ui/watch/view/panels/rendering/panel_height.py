# src/rl_fzerox/ui/watch/view/panels/rendering/panel_height.py
from __future__ import annotations

from rl_fzerox.ui.watch.view.panels.visuals.viz import (
    _control_viz_height,
    _flag_viz_height,
    _wrap_text,
)
from rl_fzerox.ui.watch.view.screen.layout import LAYOUT
from rl_fzerox.ui.watch.view.screen.theme import PALETTE
from rl_fzerox.ui.watch.view.screen.types import PanelLine, PanelSection, ViewerFonts


def _column_content_height(
    fonts: ViewerFonts,
    sections: list[PanelSection],
    *,
    width: int,
) -> int:
    y = 0
    for section_index, section in enumerate(sections):
        section_title = fonts.section.render(section.title, True, PALETTE.text_primary)
        y += section_title.get_height() + LAYOUT.section_title_gap
        y += LAYOUT.section_rule_gap
        for line in section.lines:
            if line.divider:
                y += LAYOUT.line_gap + 1
                continue
            if line.label and line.wrap:
                y += _wrapped_line_height(fonts, line, width=width)
                continue
            if line.label:
                y += _inline_or_stacked_line_height(fonts, line, width=width)
            else:
                value_surface = fonts.small.render(line.value, True, line.color)
                y += value_surface.get_height() + LAYOUT.line_gap
        if section.control_viz is not None:
            y += LAYOUT.control_viz_gap + _control_viz_height(fonts)
        if section.flag_viz is not None:
            y += LAYOUT.control_viz_gap + _flag_viz_height(fonts, section.flag_viz)
        if section_index < len(sections) - 1:
            y += LAYOUT.section_gap
    return y


def _wrapped_line_height(fonts: ViewerFonts, line: PanelLine, *, width: int) -> int:
    label_surface = fonts.small.render(line.label, True, PALETTE.text_muted)
    y = label_surface.get_height() + LAYOUT.line_gap
    wrapped_lines = _wrap_text(
        fonts.small,
        line.value,
        width - LAYOUT.wrapped_value_indent,
    )
    for wrapped_line in wrapped_lines:
        value_surface = fonts.small.render(wrapped_line, True, line.color)
        y += value_surface.get_height() + LAYOUT.line_gap
    if len(wrapped_lines) >= line.min_value_lines:
        return y
    blank_height = fonts.small.render("Ag", True, PALETTE.text_primary).get_height()
    return y + (line.min_value_lines - len(wrapped_lines)) * (blank_height + LAYOUT.line_gap)


def _inline_or_stacked_line_height(fonts: ViewerFonts, line: PanelLine, *, width: int) -> int:
    label_font = fonts.record_header if line.heading else fonts.small
    value_font = fonts.small if line.heading else fonts.body
    label_surface = label_font.render(line.label, True, PALETTE.text_muted)
    value_surface = value_font.render(line.value, True, line.color)
    status_text_surface = fonts.small.render(line.status_text, True, line.color)
    value_width = (
        value_surface.get_width()
        if line.status_icon is None
        else value_surface.get_height() + status_text_surface.get_width()
    )
    inline_value_space = width - label_surface.get_width() - LAYOUT.inline_value_gap
    if value_width <= inline_value_space:
        return max(label_surface.get_height(), value_surface.get_height()) + LAYOUT.line_gap
    return label_surface.get_height() + value_surface.get_height() + (2 * LAYOUT.line_gap)
