# src/rl_fzerox/ui/watch/view/panels/draw.py
from __future__ import annotations

from rl_fzerox.ui.watch.view.components.cockpit import _draw_control_viz
from rl_fzerox.ui.watch.view.components.tokens import _draw_flag_viz
from rl_fzerox.ui.watch.view.panels.model import _build_panel_columns, _panel_column_widths
from rl_fzerox.ui.watch.view.panels.viz import _wrap_text
from rl_fzerox.ui.watch.view.screen.layout import LAYOUT
from rl_fzerox.ui.watch.view.screen.theme import PALETTE
from rl_fzerox.ui.watch.view.screen.types import PanelLine, PanelSection, ViewerFonts


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
    gas_level: float,
    thrust_warning_threshold: float | None,
    boost_active: bool,
    boost_lamp_level: float,
    policy_label: str | None,
    policy_curriculum_stage: str | None,
    policy_deterministic: bool | None,
    policy_action,
    policy_reload_age_seconds: float | None,
    policy_reload_error: str | None,
    best_finish_position: int | None,
    best_finish_times: dict[str, int],
    track_pool_records: tuple[dict[str, object], ...],
    continuous_drive_deadzone: float,
    continuous_air_brake_mode: str,
    continuous_air_brake_disabled: bool,
    action_repeat: int,
    max_episode_steps: int,
    stuck_step_limit: int | None,
    wrong_way_timer_limit: int | None,
    progress_frontier_stall_limit_frames: int | None,
    stuck_min_speed_kph: float,
    game_display_size: tuple[int, int],
    observation_shape: tuple[int, ...],
    observation_state,
    observation_state_feature_names: tuple[str, ...],
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
        gas_level=gas_level,
        thrust_warning_threshold=thrust_warning_threshold,
        boost_active=boost_active,
        boost_lamp_level=boost_lamp_level,
        policy_label=policy_label,
        policy_curriculum_stage=policy_curriculum_stage,
        policy_deterministic=policy_deterministic,
        policy_action=policy_action,
        policy_reload_age_seconds=policy_reload_age_seconds,
        policy_reload_error=policy_reload_error,
        best_finish_position=best_finish_position,
        best_finish_times=best_finish_times,
        track_pool_records=track_pool_records,
        continuous_drive_deadzone=continuous_drive_deadzone,
        continuous_air_brake_mode=continuous_air_brake_mode,
        continuous_air_brake_disabled=continuous_air_brake_disabled,
        action_repeat=action_repeat,
        max_episode_steps=max_episode_steps,
        stuck_step_limit=stuck_step_limit,
        wrong_way_timer_limit=wrong_way_timer_limit,
        progress_frontier_stall_limit_frames=progress_frontier_stall_limit_frames,
        stuck_min_speed_kph=stuck_min_speed_kph,
        game_display_size=game_display_size,
        observation_shape=observation_shape,
        observation_state=observation_state,
        observation_state_feature_names=observation_state_feature_names,
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
    left_column_width, middle_column_width, stats_column_width = _panel_column_widths(content_width)
    left_x = x
    middle_x = x + left_column_width + LAYOUT.column_gap
    stats_x = middle_x + middle_column_width + LAYOUT.column_gap
    _draw_column_separator(
        pygame=pygame,
        screen=screen,
        x=middle_x - (LAYOUT.column_gap // 2),
        y=y,
        bottom=panel_rect.bottom - LAYOUT.panel_padding,
    )
    _draw_column_separator(
        pygame=pygame,
        screen=screen,
        x=stats_x - (LAYOUT.column_gap // 2),
        y=y,
        bottom=panel_rect.bottom - LAYOUT.panel_padding,
    )

    _draw_column(
        pygame=pygame,
        screen=screen,
        fonts=fonts,
        x=left_x,
        y=y,
        width=left_column_width,
        sections=columns.left,
    )
    _draw_column(
        pygame=pygame,
        screen=screen,
        fonts=fonts,
        x=middle_x,
        y=y,
        width=middle_column_width,
        sections=columns.middle,
    )
    _draw_column(
        pygame=pygame,
        screen=screen,
        fonts=fonts,
        x=stats_x,
        y=y,
        width=stats_column_width,
        sections=columns.stats,
    )


def _draw_column_separator(*, pygame, screen, x: int, y: int, bottom: int) -> None:
    pygame.draw.line(screen, PALETTE.panel_border, (x, y), (x, bottom), width=1)


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
            y = _draw_labeled_value_line(
                pygame=pygame,
                screen=screen,
                fonts=fonts,
                x=x,
                y=y,
                width=width,
                line=line,
            )
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


def _draw_panel_divider(*, pygame, screen, x: int, y: int, width: int) -> int:
    line_y = y + max(1, LAYOUT.line_gap // 2)
    pygame.draw.line(screen, PALETTE.panel_border, (x, line_y), (x + width, line_y), width=1)
    return line_y + LAYOUT.line_gap + 1


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


def _draw_labeled_value_line(
    *,
    pygame,
    screen,
    fonts: ViewerFonts,
    x: int,
    y: int,
    width: int,
    line: PanelLine,
) -> int:
    label_font = fonts.record_header if line.heading else fonts.small
    label_color = PALETTE.text_primary if line.heading else PALETTE.text_muted
    label_surface = label_font.render(line.label, True, label_color)

    value_font = fonts.small if line.heading else fonts.body
    value_surface = value_font.render(line.value, True, line.color)
    status_text_surface = fonts.small.render(line.status_text, True, line.color)
    value_width = (
        value_surface.get_width()
        if line.status_icon is None
        else value_surface.get_height() + status_text_surface.get_width()
    )
    inline_value_space = width - label_surface.get_width() - LAYOUT.inline_value_gap
    if value_width <= inline_value_space:
        screen.blit(label_surface, (x, y))
        if line.status_icon is None:
            value_x = x + width - value_surface.get_width()
            screen.blit(value_surface, (value_x, y - 1))
        else:
            center = (
                x + width - (value_surface.get_height() // 2),
                y + (max(label_surface.get_height(), value_surface.get_height()) // 2),
            )
            if line.status_text:
                text_x = center[0] - (value_surface.get_height() // 2) - 4
                screen.blit(
                    status_text_surface,
                    (
                        text_x - status_text_surface.get_width(),
                        y + (label_surface.get_height() - status_text_surface.get_height()) // 2,
                    ),
                )
            _draw_status_icon(
                pygame,
                screen,
                icon=line.status_icon,
                color=line.color,
                center=center,
            )
        return y + max(label_surface.get_height(), value_surface.get_height()) + LAYOUT.line_gap

    screen.blit(label_surface, (x, y))
    y += label_surface.get_height() + LAYOUT.line_gap
    if line.status_icon is not None:
        center = (
            x + width - (value_surface.get_height() // 2),
            y + (value_surface.get_height() // 2),
        )
        if line.status_text:
            text_x = center[0] - (value_surface.get_height() // 2) - 4
            screen.blit(
                status_text_surface,
                (text_x - status_text_surface.get_width(), y),
            )
        _draw_status_icon(
            pygame,
            screen,
            icon=line.status_icon,
            color=line.color,
            center=center,
        )
        return y + value_surface.get_height() + LAYOUT.line_gap

    fitted_value = _fit_text(value_font, line.value, width)
    value_surface = value_font.render(fitted_value, True, line.color)
    screen.blit(value_surface, (x, y - 1))
    return y + value_surface.get_height() + LAYOUT.line_gap


def _draw_status_icon(
    pygame,
    screen,
    *,
    icon: str,
    color,
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


def _fit_text(font, text: str, max_width: int) -> str:
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
