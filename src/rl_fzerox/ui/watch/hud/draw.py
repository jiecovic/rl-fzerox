# src/rl_fzerox/ui/watch/hud/draw.py
from __future__ import annotations

from rl_fzerox.ui.watch.hud.format import _format_observation_summary
from rl_fzerox.ui.watch.hud.model import (
    _build_panel_columns,
    _observation_preview_size,
    _wrap_text,
)
from rl_fzerox.ui.watch.layout import LAYOUT, PALETTE, PanelSection, ViewerFonts
from rl_fzerox.ui.watch.render.widgets import (
    _draw_control_viz,
    _draw_flag_viz,
)


def _draw_side_panel(
    *,
    pygame,
    screen,
    fonts: ViewerFonts,
    panel_rect,
    episode: int,
    info: dict[str, object],
    reset_info: dict[str, object],
    episode_reward: float,
    paused: bool,
    control_state,
    policy_label: str | None,
    policy_action,
    policy_reload_age_seconds: float | None,
    policy_reload_error: str | None,
    action_repeat: int,
    stuck_step_limit: int,
    game_display_size: tuple[int, int],
    observation_shape: tuple[int, ...],
    observation_surface,
    telemetry,
) -> None:
    pygame.draw.rect(screen, PALETTE.panel_background, panel_rect)
    pygame.draw.line(
        screen,
        PALETTE.panel_border,
        panel_rect.topleft,
        panel_rect.bottomleft,
        width=2,
    )

    x = panel_rect.x + LAYOUT.panel_padding
    y = panel_rect.y + LAYOUT.panel_padding
    panel_width = panel_rect.width - (2 * LAYOUT.panel_padding)
    columns = _build_panel_columns(
        episode=episode,
        info=info,
        reset_info=reset_info,
        episode_reward=episode_reward,
        paused=paused,
        control_state=control_state,
        policy_label=policy_label,
        policy_action=policy_action,
        policy_reload_age_seconds=policy_reload_age_seconds,
        policy_reload_error=policy_reload_error,
        action_repeat=action_repeat,
        stuck_step_limit=stuck_step_limit,
        game_display_size=game_display_size,
        observation_shape=observation_shape,
        telemetry=telemetry,
    )

    y = _draw_panel_title(
        screen=screen,
        fonts=fonts,
        x=x,
        y=y,
        title="F-Zero X Watch",
        subtitle="live emulator session",
    )
    y += LAYOUT.title_section_gap

    content_width = panel_width
    left_column_width = (content_width - LAYOUT.column_gap) // 2
    right_column_width = content_width - LAYOUT.column_gap - left_column_width
    left_x = x
    right_x = x + left_column_width + LAYOUT.column_gap

    _draw_column(
        pygame=pygame,
        screen=screen,
        fonts=fonts,
        x=left_x,
        y=y,
        width=left_column_width,
        sections=columns.left,
    )
    right_column_bottom = _draw_column(
        pygame=pygame,
        screen=screen,
        fonts=fonts,
        x=right_x,
        y=y,
        width=right_column_width,
        sections=columns.right,
    )
    _draw_observation_preview(
        pygame=pygame,
        screen=screen,
        fonts=fonts,
        surface=observation_surface,
        x=right_x,
        y=right_column_bottom + LAYOUT.section_gap,
        width=right_column_width,
        observation_shape=observation_shape,
    )


def _draw_column(
    *,
    pygame,
    screen,
    fonts: ViewerFonts,
    x: int,
    y: int,
    width: int,
    sections: list[PanelSection],
) -> int:
    current_y = y
    for section_index, section in enumerate(sections):
        current_y = _draw_section(
            pygame=pygame,
            screen=screen,
            fonts=fonts,
            x=x,
            y=current_y,
            width=width,
            section=section,
        )
        if section_index < len(sections) - 1:
            current_y += LAYOUT.section_gap
    return current_y


def _draw_panel_title(*, screen, fonts, x: int, y: int, title: str, subtitle: str) -> int:
    title_surface = fonts.title.render(title, True, PALETTE.text_primary)
    subtitle_surface = fonts.small.render(subtitle, True, PALETTE.text_muted)
    screen.blit(title_surface, (x, y))
    subtitle_y = y + title_surface.get_height() + LAYOUT.title_gap
    screen.blit(subtitle_surface, (x, subtitle_y))
    return subtitle_y + subtitle_surface.get_height()


def _draw_section(
    *,
    pygame,
    screen,
    fonts: ViewerFonts,
    x: int,
    y: int,
    width: int,
    section: PanelSection,
) -> int:
    section_title = fonts.section.render(section.title, True, PALETTE.text_primary)
    screen.blit(section_title, (x, y))
    y += section_title.get_height() + LAYOUT.section_title_gap
    pygame.draw.line(screen, PALETTE.panel_border, (x, y), (x + width, y), width=1)
    y += LAYOUT.section_rule_gap

    for line in section.lines:
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
            label_surface = fonts.small.render(line.label, True, PALETTE.text_muted)
            screen.blit(label_surface, (x, y))
            value_surface = fonts.body.render(line.value, True, line.color)
            value_x = x + width - value_surface.get_width()
            screen.blit(value_surface, (value_x, y - 1))
            y += max(label_surface.get_height(), value_surface.get_height()) + LAYOUT.line_gap
            continue

        value_surface = fonts.small.render(line.value, True, line.color)
        screen.blit(value_surface, (x, y))
        y += value_surface.get_height() + LAYOUT.line_gap

    if section.control_viz is not None:
        y += LAYOUT.control_viz_gap
        y = _draw_control_viz(
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

    return y


def _draw_wrapped_line(
    *,
    screen,
    fonts: ViewerFonts,
    x: int,
    y: int,
    width: int,
    label: str,
    value: str,
    color,
    min_value_lines: int,
) -> int:
    label_surface = fonts.small.render(label, True, PALETTE.text_muted)
    screen.blit(label_surface, (x, y))
    y += label_surface.get_height() + LAYOUT.line_gap

    wrapped_lines = _wrap_text(
        fonts.small,
        value,
        width - LAYOUT.wrapped_value_indent,
    )
    for wrapped_line in wrapped_lines:
        value_surface = fonts.small.render(wrapped_line, True, color)
        screen.blit(value_surface, (x + LAYOUT.wrapped_value_indent, y))
        y += value_surface.get_height() + LAYOUT.line_gap

    if len(wrapped_lines) < min_value_lines:
        blank_height = fonts.small.render("Ag", True, PALETTE.text_primary).get_height()
        y += (min_value_lines - len(wrapped_lines)) * (blank_height + LAYOUT.line_gap)

    return y


def _draw_observation_preview(
    *,
    pygame,
    screen,
    fonts: ViewerFonts,
    surface,
    x: int,
    y: int,
    width: int,
    observation_shape: tuple[int, ...],
) -> None:
    preview_width, preview_height = _observation_preview_size(observation_shape)
    title_surface = fonts.section.render("Policy Obs", True, PALETTE.text_primary)
    subtitle_surface = fonts.small.render(
        _format_observation_summary(observation_shape),
        True,
        PALETTE.text_muted,
    )
    screen.blit(title_surface, (x, y))
    y += title_surface.get_height() + LAYOUT.preview_title_gap
    screen.blit(subtitle_surface, (x, y))
    y += subtitle_surface.get_height() + LAYOUT.section_rule_gap

    preview_x = x + max(0, (width - preview_width) // 2)
    preview_rect = pygame.Rect(preview_x, y, preview_width, preview_height)
    pygame.draw.rect(screen, PALETTE.app_background, preview_rect)
    screen.blit(surface, preview_rect.topleft)
    pygame.draw.rect(
        screen,
        PALETTE.text_warning,
        preview_rect,
        width=2,
        border_radius=4,
    )
