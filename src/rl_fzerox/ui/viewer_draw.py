# src/rl_fzerox/ui/viewer_draw.py
from __future__ import annotations

import numpy as np

from rl_fzerox.core.emulator.video import display_size
from rl_fzerox.ui.viewer_format import (
    _build_panel_columns,
    _display_aspect_ratio,
    _format_observation_summary,
    _observation_preview_size,
    _preview_frame,
    _preview_panel_size,
    _window_size,
    _wrap_text,
)
from rl_fzerox.ui.viewer_layout import (
    FONT_SIZES,
    LAYOUT,
    PALETTE,
    ControlViz,
    FlagViz,
    PanelSection,
    ViewerFonts,
)


def _create_fonts(pygame) -> ViewerFonts:
    return ViewerFonts(
        title=pygame.font.Font(None, FONT_SIZES.title),
        section=pygame.font.Font(None, FONT_SIZES.section),
        body=pygame.font.Font(None, FONT_SIZES.body),
        small=pygame.font.Font(None, FONT_SIZES.small),
    )


def _draw_frame(
    *,
    pygame,
    screen,
    fonts,
    raw_frame: np.ndarray,
    observation: np.ndarray,
    episode: int,
    info: dict[str, object],
    reset_info: dict[str, object],
    episode_reward: float,
    paused: bool,
    control_state,
    policy_label: str | None,
    policy_action: np.ndarray | None,
    policy_reload_age_seconds: float | None,
    policy_reload_error: str | None,
    telemetry,
) -> None:
    game_surface = _rgb_surface(pygame, raw_frame)
    game_display_size = display_size(raw_frame.shape, _display_aspect_ratio(info))
    if game_surface.get_size() != game_display_size:
        game_surface = pygame.transform.scale(game_surface, game_display_size)
    preview_frame = _preview_frame(observation)
    observation_display_size = _observation_preview_size(observation.shape)
    observation_surface = _rgb_surface(pygame, preview_frame)
    if observation_surface.get_size() != observation_display_size:
        observation_surface = pygame.transform.scale(
            observation_surface,
            observation_display_size,
        )
    preview_panel_size = _preview_panel_size(observation.shape)

    screen.fill(PALETTE.app_background)
    screen.blit(game_surface, (0, 0))
    preview_panel_origin = (
        game_display_size[0] + LAYOUT.preview_gap,
        0,
    )
    _draw_observation_preview(
        pygame=pygame,
        screen=screen,
        fonts=fonts,
        surface=observation_surface,
        panel_origin=preview_panel_origin,
        observation_shape=observation.shape,
    )
    panel_rect = pygame.Rect(
        game_display_size[0]
        + LAYOUT.preview_gap
        + preview_panel_size[0]
        + LAYOUT.preview_gap,
        0,
        LAYOUT.panel_width,
        _window_size(game_display_size, observation.shape)[1],
    )
    _draw_side_panel(
        pygame=pygame,
        screen=screen,
        fonts=fonts,
        panel_rect=panel_rect,
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
        game_display_size=game_display_size,
        observation_shape=observation.shape,
        telemetry=telemetry,
    )
    pygame.display.flip()


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
    policy_action: np.ndarray | None,
    policy_reload_age_seconds: float | None,
    policy_reload_error: str | None,
    game_display_size: tuple[int, int],
    observation_shape: tuple[int, ...],
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
    _draw_column(
        pygame=pygame,
        screen=screen,
        fonts=fonts,
        x=right_x,
        y=y,
        width=right_column_width,
        sections=columns.right,
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
) -> None:
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


def _draw_control_viz(
    *,
    pygame,
    screen,
    fonts: ViewerFonts,
    x: int,
    y: int,
    width: int,
    control_viz: ControlViz,
) -> int:
    steer_label = fonts.small.render("Steer", True, PALETTE.text_muted)
    drive_label = fonts.small.render("Drive", True, PALETTE.text_muted)
    drive_x = x + width - LAYOUT.control_drive_width - LAYOUT.control_drive_offset_x
    left_widget_width = max(
        48,
        width
        - LAYOUT.control_drive_width
        - LAYOUT.control_drive_offset_x
        - LAYOUT.control_widget_gap,
    )
    left_drift_width = _pill_width(fonts.small, "drift")
    right_drift_width = _pill_width(fonts.small, "drift")
    max_steer_width = max(
        48,
        left_widget_width
        - left_drift_width
        - right_drift_width
        - (2 * LAYOUT.control_side_pill_gap),
    )
    steer_width = min(LAYOUT.control_steer_width, max_steer_width)
    steer_group_width = (
        left_drift_width
        + LAYOUT.control_side_pill_gap
        + steer_width
        + LAYOUT.control_side_pill_gap
        + right_drift_width
    )
    steer_group_x = x + max(0, (left_widget_width - steer_group_width) // 2)
    left_drift_x = steer_group_x
    steer_x = left_drift_x + left_drift_width + LAYOUT.control_side_pill_gap
    right_drift_x = steer_x + steer_width + LAYOUT.control_side_pill_gap

    screen.blit(steer_label, (x, y))
    screen.blit(drive_label, (drive_x - 12, y))
    y += steer_label.get_height() + LAYOUT.control_track_gap

    drive_y = y
    steer_y = drive_y + (LAYOUT.control_drive_height - LAYOUT.control_steer_height) // 2
    steer_mid_y = steer_y + LAYOUT.control_steer_height // 2
    steer_mid_x = steer_x + steer_width // 2
    drift_pill_y = steer_mid_y - (_pill_height(fonts.small) // 2)

    _draw_pill(
        pygame=pygame,
        screen=screen,
        font=fonts.small,
        x=left_drift_x,
        y=drift_pill_y,
        label="drift",
        active=control_viz.drift_direction < 0,
        active_text_color=PALETTE.text_primary,
        active_fill_color=PALETTE.flag_active_background,
        active_border_color=PALETTE.flag_active_border,
    )
    _draw_pill(
        pygame=pygame,
        screen=screen,
        font=fonts.small,
        x=right_drift_x,
        y=drift_pill_y,
        label="drift",
        active=control_viz.drift_direction > 0,
        active_text_color=PALETTE.text_primary,
        active_fill_color=PALETTE.flag_active_background,
        active_border_color=PALETTE.flag_active_border,
    )

    steer_track = pygame.Rect(
        steer_x,
        steer_y,
        steer_width,
        LAYOUT.control_steer_height,
    )
    pygame.draw.rect(
        screen,
        PALETTE.control_track,
        steer_track,
        border_radius=LAYOUT.control_steer_height // 2,
    )

    steer_extent = round((steer_width // 2) * min(1.0, abs(control_viz.steer_x)))
    if steer_extent > 0:
        steer_fill = pygame.Rect(
            steer_mid_x - steer_extent if control_viz.steer_x < 0 else steer_mid_x,
            steer_y,
            steer_extent,
            LAYOUT.control_steer_height,
        )
        pygame.draw.rect(
            screen,
            PALETTE.text_primary,
            steer_fill,
            border_radius=LAYOUT.control_steer_height // 2,
        )

    steer_knob_x = steer_mid_x + round((steer_width // 2) * control_viz.steer_x)
    _draw_round_marker(
        pygame=pygame,
        screen=screen,
        color=PALETTE.control_knob,
        center=(steer_knob_x, steer_mid_y),
        radius=LAYOUT.control_marker_radius,
        outline_color=PALETTE.control_knob_outline,
    )

    drive_track = pygame.Rect(
        drive_x,
        drive_y,
        LAYOUT.control_drive_width,
        LAYOUT.control_drive_height,
    )
    pygame.draw.rect(
        screen,
        PALETTE.control_track,
        drive_track,
        border_radius=LAYOUT.control_drive_width // 2,
    )

    drive_levels = {
        1: drive_y + LAYOUT.control_marker_radius,
        0: drive_y + LAYOUT.control_drive_height // 2,
        -1: drive_y + LAYOUT.control_drive_height - LAYOUT.control_marker_radius,
    }
    inactive_radius = max(2, LAYOUT.control_marker_radius - 2)
    drive_center_x = drive_x + LAYOUT.control_drive_width // 2
    for level, marker_y in drive_levels.items():
        color = PALETTE.panel_border
        radius = inactive_radius
        if control_viz.drive_level == level:
            if level > 0:
                color = PALETTE.text_accent
            elif level < 0:
                color = PALETTE.text_warning
            else:
                color = PALETTE.control_coast
            radius = LAYOUT.control_marker_radius
        _draw_round_marker(
            pygame=pygame,
            screen=screen,
            color=color,
            center=(drive_center_x, marker_y),
            radius=radius,
            outline_color=None,
        )

    y += LAYOUT.control_drive_height + LAYOUT.control_caption_gap
    if control_viz.drive_level > 0:
        mode = "throttle"
    elif control_viz.drive_level < 0:
        mode = "brake"
    else:
        mode = "coast"
    mode_color = (
        PALETTE.text_accent
        if control_viz.drive_level > 0
        else PALETTE.text_warning
        if control_viz.drive_level < 0
        else PALETTE.text_muted
    )
    mode_surface = fonts.small.render(mode, True, mode_color)
    mode_x = drive_center_x - (mode_surface.get_width() // 2)
    screen.blit(mode_surface, (mode_x, y))
    y += mode_surface.get_height() + LAYOUT.control_boost_gap
    boost_x = drive_center_x - (_pill_width(fonts.small, "boost") // 2)
    _draw_pill(
        pygame=pygame,
        screen=screen,
        font=fonts.small,
        x=boost_x,
        y=y,
        label="boost",
        active=control_viz.boost_pressed,
        active_text_color=PALETTE.text_primary,
        active_fill_color=(88, 66, 28),
        active_border_color=PALETTE.text_warning,
    )
    return y + _pill_height(fonts.small)


def _draw_round_marker(
    *,
    pygame,
    screen,
    color,
    center: tuple[int, int],
    radius: int,
    outline_color,
) -> None:
    gfxdraw = getattr(pygame, "gfxdraw", None)
    if gfxdraw is not None:
        gfxdraw.filled_circle(screen, center[0], center[1], radius, color)
        gfxdraw.aacircle(screen, center[0], center[1], radius, color)
        if outline_color is not None and radius > 1:
            gfxdraw.aacircle(screen, center[0], center[1], radius, outline_color)
        return

    pygame.draw.circle(screen, color, center, radius)
    if outline_color is not None and radius > 1:
        pygame.draw.circle(screen, outline_color, center, radius, width=1)


def _draw_flag_viz(
    *,
    pygame,
    screen,
    fonts: ViewerFonts,
    x: int,
    y: int,
    flag_viz: FlagViz,
) -> int:
    label_surface = fonts.small.render("Flags", True, PALETTE.text_muted)
    screen.blit(label_surface, (x, y))
    y += label_surface.get_height() + LAYOUT.line_gap

    pill_height = 0
    for row in flag_viz.rows:
        row_x = x
        for token in row:
            pill_rect = _draw_pill(
                pygame=pygame,
                screen=screen,
                font=fonts.small,
                x=row_x,
                y=y,
                label=token.label,
                active=token.active,
                active_text_color=PALETTE.text_primary,
                active_fill_color=PALETTE.flag_active_background,
                active_border_color=PALETTE.flag_active_border,
            )
            pill_height = pill_rect.height
            row_x += pill_rect.width + LAYOUT.flag_token_gap
        y += pill_height + LAYOUT.line_gap

    return y


def _pill_width(font, label: str) -> int:
    return font.render(label, True, PALETTE.text_primary).get_width() + (
        2 * LAYOUT.flag_token_pad_x
    )


def _pill_height(font) -> int:
    return font.render("pill", True, PALETTE.text_primary).get_height() + (
        2 * LAYOUT.flag_token_pad_y
    )


def _draw_pill(
    *,
    pygame,
    screen,
    font,
    x: int,
    y: int,
    label: str,
    active: bool,
    active_text_color,
    active_fill_color,
    active_border_color,
):
    text_color = active_text_color if active else PALETTE.text_muted
    fill_color = active_fill_color if active else PALETTE.flag_inactive_background
    border_color = active_border_color if active else PALETTE.flag_inactive_border
    token_surface = font.render(label, True, text_color)
    pill_rect = pygame.Rect(
        x,
        y,
        token_surface.get_width() + (2 * LAYOUT.flag_token_pad_x),
        token_surface.get_height() + (2 * LAYOUT.flag_token_pad_y),
    )
    pygame.draw.rect(
        screen,
        fill_color,
        pill_rect,
        border_radius=8,
    )
    pygame.draw.rect(
        screen,
        border_color,
        pill_rect,
        width=1,
        border_radius=8,
    )
    screen.blit(
        token_surface,
        (
            pill_rect.x + LAYOUT.flag_token_pad_x,
            pill_rect.y + LAYOUT.flag_token_pad_y,
        ),
    )
    return pill_rect


def _rgb_surface(pygame, frame: np.ndarray):
    rgb_frame = np.ascontiguousarray(frame)
    height, width, channels = rgb_frame.shape
    if channels != 3:
        raise ValueError(f"Expected an RGB frame for display, got shape {frame.shape!r}")
    return pygame.image.frombuffer(rgb_frame.tobytes(), (width, height), "RGB")


def _draw_observation_preview(
    *,
    pygame,
    screen,
    fonts: ViewerFonts,
    surface,
    panel_origin: tuple[int, int],
    observation_shape: tuple[int, ...],
) -> None:
    preview_width, preview_height = _observation_preview_size(observation_shape)
    panel_width, panel_height = _preview_panel_size(observation_shape)
    panel_rect = pygame.Rect(
        panel_origin[0],
        panel_origin[1],
        panel_width,
        panel_height,
    )
    pygame.draw.rect(screen, PALETTE.panel_background, panel_rect)
    pygame.draw.line(
        screen,
        PALETTE.panel_border,
        panel_rect.topleft,
        panel_rect.bottomleft,
        width=2,
    )

    x = panel_rect.x + LAYOUT.preview_padding
    y = panel_rect.y + LAYOUT.preview_padding
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

    preview_rect = pygame.Rect(x, y, preview_width, preview_height)
    pygame.draw.rect(screen, PALETTE.app_background, preview_rect)
    screen.blit(surface, preview_rect.topleft)
