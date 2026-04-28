# src/rl_fzerox/ui/watch/view/panels/core/lines.py
from __future__ import annotations

from rl_fzerox.ui.watch.view.screen.theme import PALETTE, Color
from rl_fzerox.ui.watch.view.screen.types import PanelLine, StatusIcon


def panel_line(
    label: str,
    value: str,
    color: Color,
    *,
    wrap: bool = False,
    min_value_lines: int = 1,
    divider: bool = False,
    heading: bool = False,
    status_icon: StatusIcon | None = None,
    status_text: str = "",
    label_color: Color | None = None,
    click_course_id: str | None = None,
) -> PanelLine:
    return PanelLine(
        label=label,
        value=value,
        color=color,
        wrap=wrap,
        min_value_lines=min_value_lines,
        divider=divider,
        heading=heading,
        status_icon=status_icon,
        status_text=status_text,
        label_color=label_color,
        click_course_id=click_course_id,
    )


def panel_divider() -> PanelLine:
    return panel_line("", "", PALETTE.panel_border, divider=True)


def panel_heading(label: str) -> PanelLine:
    return panel_line(label, "", PALETTE.text_primary, heading=True)
