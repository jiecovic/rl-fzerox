# src/rl_fzerox/ui/watch/render/frame.py
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from fzerox_emulator.arrays import ObservationFrame, RgbFrame, StateVector
from rl_fzerox.core.envs.actions import ActionValue
from rl_fzerox.ui.watch.hud.draw import _draw_side_panel
from rl_fzerox.ui.watch.hud.format import _format_observation_summary
from rl_fzerox.ui.watch.hud.model import (
    _observation_preview_size,
    _preview_frame,
    _window_size,
)
from rl_fzerox.ui.watch.layout import FONT_SIZES, LAYOUT, PALETTE, Color, ViewerFonts


@dataclass(frozen=True)
class _GlassViewStyle:
    """Visual treatment for the main game viewport in watch mode."""

    frame_x: int = 14
    frame_y: int = 12
    frame_radius: int = 20
    viewport_radius: int = 12
    frame_fill: Color = (9, 12, 15)
    frame_edge: Color = (40, 50, 62)
    frame_highlight: Color = (86, 101, 116)
    screen_lip: Color = (3, 5, 6)
    glass_edge: Color = (62, 116, 96)
    shadow: Color = (2, 3, 4)


_GLASS_VIEW_STYLE = _GlassViewStyle()
_GLASS_OVERLAY_CACHE: dict[tuple[int, int, int], object] = {}
_GLASS_MASK_CACHE: dict[tuple[int, int, int], object] = {}


def _create_fonts(pygame) -> ViewerFonts:
    return ViewerFonts(
        title=pygame.font.Font(None, FONT_SIZES.title),
        section=pygame.font.Font(None, FONT_SIZES.section),
        body=_create_mono_font(pygame, FONT_SIZES.body),
        small=pygame.font.Font(None, FONT_SIZES.small),
    )


def _create_mono_font(pygame, size: int):
    font_path = pygame.font.match_font(
        ("dejavusansmono", "liberationmono", "consolas", "couriernew", "monospace")
    )
    if font_path is not None:
        return pygame.font.Font(font_path, size)
    return pygame.font.SysFont("monospace", size)


def _create_screen(
    pygame,
    game_display_size: tuple[int, int],
    observation_shape: tuple[int, ...],
):
    _apply_window_position_hint()
    screen = pygame.display.set_mode(_window_size(game_display_size, observation_shape))
    pygame.display.set_caption("F-Zero X Watch")
    return screen


def _ensure_screen(
    pygame,
    screen,
    game_display_size: tuple[int, int],
    observation_shape: tuple[int, ...],
):
    if screen is None:
        return _create_screen(pygame, game_display_size, observation_shape)
    if screen.get_size() == _window_size(game_display_size, observation_shape):
        return screen
    return _create_screen(pygame, game_display_size, observation_shape)


def _watch_game_display_size() -> tuple[int, int]:
    return LAYOUT.game_display_size


def _draw_frame(
    *,
    pygame,
    screen,
    fonts,
    raw_frame: RgbFrame,
    observation: ObservationFrame,
    observation_state: StateVector | None,
    observation_state_feature_names: tuple[str, ...],
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
    policy_action: ActionValue | None,
    policy_reload_age_seconds: float | None,
    policy_reload_error: str | None,
    best_finish_position: int | None,
    continuous_drive_deadzone: float,
    continuous_air_brake_mode: str,
    continuous_air_brake_disabled: bool,
    action_repeat: int,
    max_episode_steps: int,
    stuck_step_limit: int | None,
    wrong_way_timer_limit: int | None,
    progress_frontier_stall_limit_frames: int | None,
    stuck_min_speed_kph: float,
    telemetry,
) -> None:
    game_display_size = _watch_game_display_size()
    game_surface = _rgb_surface(pygame, raw_frame)

    preview_frame = _preview_frame(observation)
    observation_display_size = _observation_preview_size(observation.shape)
    observation_surface = _rgb_surface(pygame, preview_frame)
    if observation_surface.get_size() != observation_display_size:
        raise RuntimeError(
            "Native observation preview size did not match the expected preview size: "
            f"frame={observation_surface.get_size()}, expected={observation_display_size}"
        )

    screen.fill(PALETTE.app_background)
    _draw_glass_game_view(
        pygame=pygame,
        screen=screen,
        surface=game_surface,
        outer_size=game_display_size,
    )
    _draw_observation_preview_below_game(
        pygame=pygame,
        screen=screen,
        fonts=fonts,
        surface=observation_surface,
        game_display_size=game_display_size,
        observation_shape=observation.shape,
    )
    panel_rect = pygame.Rect(
        game_display_size[0] + LAYOUT.preview_gap,
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
        observation_shape=observation.shape,
        observation_state=observation_state,
        observation_state_feature_names=observation_state_feature_names,
        telemetry=telemetry,
    )
    pygame.display.flip()


def _rgb_surface(pygame, frame: RgbFrame):
    rgb_frame = np.ascontiguousarray(frame)
    height, width, channels = rgb_frame.shape
    if channels != 3:
        raise ValueError(f"Expected an RGB frame for display, got shape {frame.shape!r}")
    return pygame.image.frombuffer(rgb_frame.tobytes(), (width, height), "RGB")


def _draw_glass_game_view(
    *,
    pygame,
    screen,
    surface,
    outer_size: tuple[int, int],
) -> None:
    style = _GLASS_VIEW_STYLE
    outer_rect = pygame.Rect(0, 0, *outer_size)
    viewport_rect = outer_rect.inflate(-(2 * style.frame_x), -(2 * style.frame_y))

    pygame.draw.rect(screen, style.shadow, outer_rect.move(0, 5), border_radius=style.frame_radius)
    pygame.draw.rect(screen, style.frame_fill, outer_rect, border_radius=style.frame_radius)
    pygame.draw.rect(
        screen,
        style.frame_highlight,
        outer_rect,
        width=2,
        border_radius=style.frame_radius,
    )
    pygame.draw.rect(
        screen,
        style.frame_edge,
        outer_rect.inflate(-6, -6),
        width=2,
        border_radius=style.frame_radius - 4,
    )

    pygame.draw.rect(
        screen,
        style.screen_lip,
        viewport_rect.inflate(8, 8),
        border_radius=style.viewport_radius + 6,
    )
    pygame.draw.rect(
        screen,
        (0, 0, 0),
        viewport_rect.inflate(2, 2),
        border_radius=style.viewport_radius + 2,
    )

    surface = _rounded_game_surface(
        pygame=pygame,
        surface=surface,
        size=viewport_rect.size,
        radius=style.viewport_radius,
    )
    screen.blit(surface, viewport_rect.topleft)
    screen.blit(
        _glass_overlay_surface(pygame, viewport_rect.size, style.viewport_radius),
        viewport_rect.topleft,
    )

    pygame.draw.rect(
        screen,
        style.glass_edge,
        viewport_rect,
        width=2,
        border_radius=style.viewport_radius,
    )
    pygame.draw.rect(
        screen,
        style.frame_highlight,
        viewport_rect.inflate(2, 2),
        width=1,
        border_radius=style.viewport_radius + 2,
    )
    pygame.draw.line(
        screen,
        (128, 142, 154),
        (outer_rect.left + 24, outer_rect.top + 7),
        (outer_rect.right - 34, outer_rect.top + 7),
    )
    pygame.draw.line(
        screen,
        (0, 0, 0),
        (outer_rect.left + 20, outer_rect.bottom - 8),
        (outer_rect.right - 26, outer_rect.bottom - 8),
    )


def _rounded_game_surface(pygame, surface, size: tuple[int, int], radius: int):
    if surface.get_size() != size:
        surface = pygame.transform.scale(surface, size)

    clipped = pygame.Surface(size, pygame.SRCALPHA)
    clipped.blit(surface, (0, 0))
    clipped.blit(
        _rounded_alpha_mask(pygame, size, radius),
        (0, 0),
        special_flags=pygame.BLEND_RGBA_MULT,
    )
    return clipped


def _rounded_alpha_mask(pygame, size: tuple[int, int], radius: int):
    cache_key = (*size, radius)
    cached = _GLASS_MASK_CACHE.get(cache_key)
    if cached is not None:
        return cached

    mask = pygame.Surface(size, pygame.SRCALPHA)
    pygame.draw.rect(mask, (255, 255, 255, 255), pygame.Rect(0, 0, *size), border_radius=radius)
    _GLASS_MASK_CACHE[cache_key] = mask
    return mask


def _glass_overlay_surface(pygame, size: tuple[int, int], radius: int):
    cache_key = (*size, radius)
    cached = _GLASS_OVERLAY_CACHE.get(cache_key)
    if cached is not None:
        return cached

    # The glass overlay only depends on viewport size; cache it instead of
    # redrawing translucent highlights every watch frame.
    width, height = size
    overlay = pygame.Surface(size, pygame.SRCALPHA)

    _draw_glass_vignette(pygame=pygame, overlay=overlay, width=width, height=height, radius=radius)
    pygame.draw.rect(
        overlay,
        (255, 255, 255, 28),
        pygame.Rect(8, 5, max(1, width - 16), max(2, height // 8)),
        border_radius=max(4, radius - 4),
    )
    pygame.draw.line(
        overlay,
        (255, 255, 255, 54),
        (radius, 4),
        (width - radius, 4),
        width=1,
    )
    pygame.draw.line(
        overlay,
        (0, 0, 0, 48),
        (radius, height - 4),
        (width - radius, height - 4),
        width=1,
    )

    overlay.blit(
        _rounded_alpha_mask(pygame, size, radius),
        (0, 0),
        special_flags=pygame.BLEND_RGBA_MULT,
    )
    _GLASS_OVERLAY_CACHE[cache_key] = overlay
    return overlay


def _draw_glass_vignette(*, pygame, overlay, width: int, height: int, radius: int) -> None:
    max_inset = min(28, width // 9, height // 9)
    for inset in range(max_inset):
        alpha = round(30 * ((max_inset - inset) / max_inset) ** 1.8)
        rect = pygame.Rect(inset, inset, width - (2 * inset), height - (2 * inset))
        pygame.draw.rect(overlay, (0, 0, 0, alpha), rect, width=1, border_radius=radius)


def _apply_window_position_hint() -> None:
    os.environ["SDL_VIDEO_WINDOW_POS"] = "100,100"


def _draw_observation_preview_below_game(
    *,
    pygame,
    screen,
    fonts: ViewerFonts,
    surface,
    game_display_size: tuple[int, int],
    observation_shape: tuple[int, ...],
) -> None:
    x = LAYOUT.preview_padding
    y = game_display_size[1] + LAYOUT.preview_gap
    width = game_display_size[0] - (2 * LAYOUT.preview_padding)
    height = screen.get_height() - y - LAYOUT.preview_padding
    if width <= 0 or height <= 0:
        return

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

    preview_width, preview_height = surface.get_size()
    available_height = screen.get_height() - y - LAYOUT.preview_padding
    scale = min(width // preview_width, available_height // preview_height)
    if scale <= 0:
        return

    scaled_size = (preview_width * scale, preview_height * scale)
    preview_x = x + max(0, (width - scaled_size[0]) // 2)
    preview_rect = pygame.Rect(preview_x, y, *scaled_size)
    pygame.draw.rect(screen, PALETTE.panel_background, preview_rect)
    preview_surface = pygame.transform.scale(surface, scaled_size)
    screen.blit(preview_surface, preview_rect.topleft)
    pygame.draw.rect(
        screen,
        PALETTE.text_warning,
        preview_rect,
        width=2,
        border_radius=4,
    )
