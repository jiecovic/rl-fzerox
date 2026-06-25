# src/rl_fzerox/ui/watch/view/panels/rendering/section_renderer.py
from __future__ import annotations

from dataclasses import dataclass

from rl_fzerox.ui.watch.view.components.cockpit.panel import _draw_control_viz
from rl_fzerox.ui.watch.view.components.tokens import _draw_flag_viz
from rl_fzerox.ui.watch.view.panels.rendering.text import (
    _centered_text_y,
    _fit_text,
    _font_line_height,
)
from rl_fzerox.ui.watch.view.panels.visuals.viz import _wrap_text
from rl_fzerox.ui.watch.view.screen.layout import LAYOUT
from rl_fzerox.ui.watch.view.screen.theme import PALETTE, Color
from rl_fzerox.ui.watch.view.screen.types import (
    PanelLine,
    PanelSection,
    PygameModule,
    PygameSurface,
    RecordCourseHitbox,
    StateFeatureHitbox,
    ViewerFonts,
)


@dataclass(frozen=True, slots=True)
class _ColumnHitboxes:
    record_courses: tuple[RecordCourseHitbox, ...] = ()
    state_features: tuple[StateFeatureHitbox, ...] = ()


def _draw_column(
    *,
    pygame: PygameModule,
    screen: PygameSurface,
    fonts: ViewerFonts,
    x: int,
    y: int,
    width: int,
    sections: list[PanelSection],
) -> tuple[int, _ColumnHitboxes]:
    current_y = y
    record_course_hitboxes: list[RecordCourseHitbox] = []
    state_feature_hitboxes: list[StateFeatureHitbox] = []
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
        record_course_hitboxes.extend(section_hitboxes.record_courses)
        state_feature_hitboxes.extend(section_hitboxes.state_features)
        if section_index < len(sections) - 1:
            current_y += LAYOUT.section_gap
    return current_y, _ColumnHitboxes(
        record_courses=tuple(record_course_hitboxes),
        state_features=tuple(state_feature_hitboxes),
    )


def _draw_section(
    *,
    pygame: PygameModule,
    screen: PygameSurface,
    fonts: ViewerFonts,
    x: int,
    y: int,
    width: int,
    section: PanelSection,
) -> tuple[int, _ColumnHitboxes]:
    record_course_hitboxes: list[RecordCourseHitbox] = []
    state_feature_hitboxes: list[StateFeatureHitbox] = []
    section_title = fonts.section.render(section.title, True, PALETTE.text_primary)
    screen.blit(section_title, (x, y))
    y += section_title.get_height() + LAYOUT.section_title_gap
    pygame.draw.line(screen, PALETTE.panel_border, (x, y), (x + width, y), width=1)
    y += LAYOUT.section_rule_gap
    label_column_width = max(
        (
            fonts.small.render(line.label, True, PALETTE.text_muted).get_width()
            for line in section.lines
            if line.label and not line.divider and not line.heading
        ),
        default=0,
    )

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
            y, record_hitbox, state_feature_hitbox = _draw_labeled_value_line(
                pygame=pygame,
                screen=screen,
                fonts=fonts,
                x=x,
                y=y,
                width=width,
                line=line,
                label_column_width=label_column_width,
            )
            if record_hitbox is not None:
                record_course_hitboxes.append(record_hitbox)
            if state_feature_hitbox is not None:
                state_feature_hitboxes.append(state_feature_hitbox)
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

    return y, _ColumnHitboxes(
        record_courses=tuple(record_course_hitboxes),
        state_features=tuple(state_feature_hitboxes),
    )


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
    label_column_width: int = 0,
) -> tuple[int, RecordCourseHitbox | None, StateFeatureHitbox | None]:
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
    state_feature_hitbox = _state_feature_hitbox(
        line=line,
        x=x,
        y=y,
        width=width,
        row_height=row_height,
    )

    screen.blit(label_surface, (x, _centered_text_y(y, row_height, label_surface)))

    effective_label_width = (
        label_surface.get_width()
        if line.heading
        else max(label_surface.get_width(), label_column_width)
    )
    inline_value_space = max(0, width - effective_label_width - LAYOUT.inline_value_gap)

    if line.status_icon is not None:
        icon_slot_width = row_height
        status_gap = max(1, LAYOUT.inline_value_gap // 2)
        fitted_status_text = ""
        accessory_width = icon_slot_width
        if line.status_text:
            status_text_space = max(0, inline_value_space - icon_slot_width - status_gap)
            fitted_status_text = _fit_text(fonts.small, line.status_text, status_text_space)
            status_text_surface = fonts.small.render(fitted_status_text, True, line.color)
            if fitted_status_text:
                accessory_width += status_gap + status_text_surface.get_width()

        value_gap = LAYOUT.inline_value_gap if line.value and accessory_width > 0 else 0
        value_space = max(0, inline_value_space - accessory_width - value_gap)
        fitted_value = _fit_text(value_font, line.value, value_space)
        value_surface = value_font.render(fitted_value, True, line.color)

        cursor_right = x + width
        center = (
            cursor_right - (icon_slot_width // 2),
            y + (row_height // 2),
        )
        _draw_status_icon(
            pygame,
            screen,
            icon=line.status_icon,
            color=line.color,
            center=center,
        )
        cursor_right -= icon_slot_width

        if fitted_status_text:
            cursor_right -= status_gap
            screen.blit(
                status_text_surface,
                (
                    cursor_right - status_text_surface.get_width(),
                    _centered_text_y(y, row_height, status_text_surface),
                ),
            )
            cursor_right -= status_text_surface.get_width()

        if fitted_value:
            cursor_right -= value_gap
            screen.blit(
                value_surface,
                (
                    cursor_right - value_surface.get_width(),
                    _centered_text_y(y, row_height, value_surface),
                ),
            )
        return y + row_height + LAYOUT.line_gap, hitbox, state_feature_hitbox

    fitted_value = _fit_text(value_font, line.value, inline_value_space)
    value_surface = value_font.render(fitted_value, True, line.color)
    value_x = x + width - value_surface.get_width()
    screen.blit(value_surface, (value_x, _centered_text_y(y, row_height, value_surface)))
    return y + row_height + LAYOUT.line_gap, hitbox, state_feature_hitbox


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


def _state_feature_hitbox(
    *,
    line: PanelLine,
    x: int,
    y: int,
    width: int,
    row_height: int,
) -> StateFeatureHitbox | None:
    if line.click_state_feature_name is None:
        return None
    return StateFeatureHitbox(
        rect=(x, y, width, row_height + LAYOUT.line_gap),
        feature_name=line.click_state_feature_name,
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
        return
    if icon == "toggle_on":
        square = pygame.Rect(x - 5, y - 5, 10, 10)
        pygame.draw.rect(screen, color, square, border_radius=2)
        pygame.draw.line(screen, PALETTE.panel_background, (x - 2, y), (x, y + 2), width=2)
        pygame.draw.line(screen, PALETTE.panel_background, (x, y + 2), (x + 3, y - 2), width=2)
        return
    if icon == "toggle_off":
        square = pygame.Rect(x - 5, y - 5, 10, 10)
        pygame.draw.rect(screen, color, square, width=1, border_radius=2)
