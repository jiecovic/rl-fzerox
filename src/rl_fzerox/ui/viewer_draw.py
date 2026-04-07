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
)
from rl_fzerox.ui.viewer_layout import FONT_SIZES, LAYOUT, PALETTE, PanelSection, ViewerFonts


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

    return y


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
