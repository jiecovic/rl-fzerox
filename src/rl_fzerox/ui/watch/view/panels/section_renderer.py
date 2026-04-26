# src/rl_fzerox/ui/watch/view/panels/section_renderer.py
from __future__ import annotations

from rl_fzerox.ui.watch.view.components.cockpit import _draw_control_viz
from rl_fzerox.ui.watch.view.components.tokens import _draw_flag_viz
from rl_fzerox.ui.watch.view.panels.text import (
    _centered_text_y,
    _fit_text,
    _font_line_height,
)
from rl_fzerox.ui.watch.view.panels.viz import _wrap_text
from rl_fzerox.ui.watch.view.screen.layout import LAYOUT
from rl_fzerox.ui.watch.view.screen.theme import PALETTE, Color
from rl_fzerox.ui.watch.view.screen.types import (
    PanelLine,
    PanelSection,
    PygameModule,
    PygameSurface,
    RecordCourseHitbox,
    ViewerFonts,
)


def _draw_column(
    *,
    pygame: PygameModule,
    screen: PygameSurface,
    fonts: ViewerFonts,
    x: int,
    y: int,
    width: int,
    sections: list[PanelSection],
) -> tuple[int, tuple[RecordCourseHitbox, ...]]:
    current_y = y
    record_course_hitboxes: list[RecordCourseHitbox] = []
    for section_index, section in enumerate(sections):
        current_y, section_hitboxes = _draw_section(
            pygame=pygame,
            screen=screen,
            fonts=fonts,
            x=x,
            y=current_y,
            width=width,
            section=section,
        )
        record_course_hitboxes.extend(section_hitboxes)
        if section_index < len(sections) - 1:
            current_y += LAYOUT.section_gap
    return current_y, tuple(record_course_hitboxes)


def _draw_section(
    *,
    pygame: PygameModule,
    screen: PygameSurface,
    fonts: ViewerFonts,
    x: int,
    y: int,
    width: int,
    section: PanelSection,
) -> tuple[int, tuple[RecordCourseHitbox, ...]]:
    record_course_hitboxes: list[RecordCourseHitbox] = []
    section_title = fonts.section.render(section.title, True, PALETTE.text_primary)
    screen.blit(section_title, (x, y))
    y += section_title.get_height() + LAYOUT.section_title_gap
    pygame.draw.line(screen, PALETTE.panel_border, (x, y), (x + width, y), width=1)
    y += LAYOUT.section_rule_gap

    for line in section.lines:
        if line.divider:
            y = _draw_panel_divider(pygame=pygame, screen=screen, x=x, y=y, width=width)
            continue
        if line.label and line.wrap:
            y = _draw_wrapped_line(
                screen=screen,
                fonts=fonts,
                x=x,
                y=y,
                width=width,
                label=line.label,
                value=line.value,
                color=line.color,
                min_value_lines=line.min_value_lines,
            )
            continue
        if line.label:
            y, hitbox = _draw_labeled_value_line(
                pygame=pygame,
                screen=screen,
                fonts=fonts,
                x=x,
                y=y,
                width=width,
                line=line,
            )
            if hitbox is not None:
                record_course_hitboxes.append(hitbox)
            continue

        value_surface = fonts.small.render(line.value, True, line.color)
        line_height = _font_line_height(fonts.small)
        screen.blit(value_surface, (x, _centered_text_y(y, line_height, value_surface)))
        y += line_height + LAYOUT.line_gap

    if section.control_viz is not None:
        y += LAYOUT.control_viz_gap
        y, _ = _draw_control_viz(
            pygame=pygame,
            screen=screen,
            fonts=fonts,
            x=x,
            y=y,
            width=width,
            control_viz=section.control_viz,
        )
    if section.flag_viz is not None:
        y += LAYOUT.control_viz_gap
        y = _draw_flag_viz(
            pygame=pygame,
            screen=screen,
            fonts=fonts,
            x=x,
            y=y,
            flag_viz=section.flag_viz,
        )

    return y, tuple(record_course_hitboxes)


def _draw_panel_divider(
    *,
    pygame: PygameModule,
    screen: PygameSurface,
    x: int,
    y: int,
    width: int,
) -> int:
    line_y = y + max(1, LAYOUT.line_gap // 2)
    pygame.draw.line(screen, PALETTE.panel_border, (x, line_y), (x + width, line_y), width=1)
    return line_y + LAYOUT.line_gap + 1


def _draw_wrapped_line(
    *,
    screen: PygameSurface,
    fonts: ViewerFonts,
    x: int,
    y: int,
    width: int,
    label: str,
    value: str,
    color: Color,
    min_value_lines: int,
) -> int:
    label_surface = fonts.small.render(label, True, PALETTE.text_muted)
    label_height = _font_line_height(fonts.small)
    screen.blit(label_surface, (x, _centered_text_y(y, label_height, label_surface)))
    y += label_height + LAYOUT.line_gap

    wrapped_lines = _wrap_text(
        fonts.small,
        value,
        width - LAYOUT.wrapped_value_indent,
    )
    for wrapped_line in wrapped_lines:
        value_surface = fonts.small.render(wrapped_line, True, color)
        value_height = _font_line_height(fonts.small)
        screen.blit(
            value_surface,
            (x + LAYOUT.wrapped_value_indent, _centered_text_y(y, value_height, value_surface)),
        )
        y += value_height + LAYOUT.line_gap

    if len(wrapped_lines) < min_value_lines:
        blank_height = _font_line_height(fonts.small)
        y += (min_value_lines - len(wrapped_lines)) * (blank_height + LAYOUT.line_gap)

    return y


def _draw_labeled_value_line(
    *,
    pygame: PygameModule,
    screen: PygameSurface,
    fonts: ViewerFonts,
    x: int,
    y: int,
    width: int,
    line: PanelLine,
) -> tuple[int, RecordCourseHitbox | None]:
    label_font = fonts.record_header if line.heading else fonts.small
    label_color = line.label_color or (PALETTE.text_primary if line.heading else PALETTE.text_muted)
    label_surface = label_font.render(line.label, True, label_color)
    label_height = _font_line_height(label_font)

    value_font = fonts.small if line.heading else fonts.body
    value_height = _font_line_height(value_font)
    status_text_surface = fonts.small.render(line.status_text, True, line.color)
    status_text_height = _font_line_height(fonts.small)
    row_height = max(label_height, value_height, status_text_height)
    hitbox = _record_course_hitbox(
        line=line,
        x=x,
        y=y,
        width=width,
        row_height=row_height,
    )

    screen.blit(label_surface, (x, _centered_text_y(y, row_height, label_surface)))

    inline_value_space = max(
        0,
        width - label_surface.get_width() - LAYOUT.inline_value_gap,
    )

    if line.status_icon is not None:
        icon_slot_width = row_height
        center = (
            x + width - (icon_slot_width // 2),
            y + (row_height // 2),
        )
        if line.status_text:
            status_gap = max(1, LAYOUT.inline_value_gap // 2)
            status_text_space = max(0, inline_value_space - icon_slot_width - status_gap)
            fitted_status_text = _fit_text(fonts.small, line.status_text, status_text_space)
            status_text_surface = fonts.small.render(fitted_status_text, True, line.color)
            text_x = center[0] - (icon_slot_width // 2) - status_gap
            screen.blit(
                status_text_surface,
                (
                    text_x - status_text_surface.get_width(),
                    _centered_text_y(y, row_height, status_text_surface),
                ),
            )
        _draw_status_icon(
            pygame,
            screen,
            icon=line.status_icon,
            color=line.color,
            center=center,
        )
        return y + row_height + LAYOUT.line_gap, hitbox

    fitted_value = _fit_text(value_font, line.value, inline_value_space)
    value_surface = value_font.render(fitted_value, True, line.color)
    value_x = x + width - value_surface.get_width()
    screen.blit(value_surface, (value_x, _centered_text_y(y, row_height, value_surface)))
    return y + row_height + LAYOUT.line_gap, hitbox


def _record_course_hitbox(
    *,
    line: PanelLine,
    x: int,
    y: int,
    width: int,
    row_height: int,
) -> RecordCourseHitbox | None:
    if line.click_course_id is None:
        return None
    return RecordCourseHitbox(
        rect=(x, y, width, row_height + LAYOUT.line_gap),
        course_id=line.click_course_id,
    )


def _draw_status_icon(
    pygame: PygameModule,
    screen: PygameSurface,
    *,
    icon: str,
    color: Color,
    center: tuple[int, int],
) -> None:
    x, y = center
    if icon == "none":
        pygame.draw.circle(screen, color, center, 4, width=1)
        return
    if icon == "in_range":
        pygame.draw.line(screen, color, (x - 5, y), (x - 2, y + 3), width=2)
        pygame.draw.line(screen, color, (x - 2, y + 3), (x + 5, y - 4), width=2)
        return
    if icon == "outside":
        triangle = ((x, y - 5), (x - 5, y + 4), (x + 5, y + 4))
        pygame.draw.polygon(screen, color, triangle, width=1)
        pygame.draw.line(screen, color, (x, y - 2), (x, y + 1), width=1)
        pygame.draw.circle(screen, color, (x, y + 3), 1)
