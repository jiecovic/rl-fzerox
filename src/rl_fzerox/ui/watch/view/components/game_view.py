# src/rl_fzerox/ui/watch/view/components/game_view.py
from __future__ import annotations

from dataclasses import dataclass

from rl_fzerox.ui.watch.view.screen.theme import Color
from rl_fzerox.ui.watch.view.screen.types import (
    PygameModule,
    PygameRect,
    PygameSurface,
    ViewerFonts,
)


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
_GLASS_OVERLAY_CACHE: dict[tuple[int, int, int], PygameSurface] = {}
_GLASS_MASK_CACHE: dict[tuple[int, int, int], PygameSurface] = {}


def _draw_glass_game_view(
    *,
    pygame: PygameModule,
    screen: PygameSurface,
    fonts: ViewerFonts,
    surface: PygameSurface,
    outer_size: tuple[int, int],
    course_label: str | None = None,
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
    if course_label:
        _draw_course_overlay(
            pygame=pygame,
            screen=screen,
            fonts=fonts,
            viewport_rect=viewport_rect,
            label=course_label,
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


def _draw_course_overlay(
    *,
    pygame: PygameModule,
    screen: PygameSurface,
    fonts: ViewerFonts,
    viewport_rect: PygameRect,
    label: str,
) -> None:
    text_color = (108, 255, 178)
    border_color = (62, 224, 154)
    label_surface = fonts.small.render(label, True, text_color)
    pad_x = 8
    pad_y = 5
    overlay_width = label_surface.get_width() + (2 * pad_x)
    overlay_height = label_surface.get_height() + (2 * pad_y)
    overlay_rect = pygame.Rect(
        viewport_rect.left + 12,
        viewport_rect.bottom - overlay_height - 12,
        overlay_width,
        overlay_height,
    )
    shadow_rect = overlay_rect.move(0, 2)
    pygame.draw.rect(screen, (0, 0, 0), shadow_rect, border_radius=7)

    overlay = pygame.Surface(overlay_rect.size, pygame.SRCALPHA)
    pygame.draw.rect(
        overlay,
        (5, 18, 13, 184),
        pygame.Rect(0, 0, *overlay_rect.size),
        border_radius=7,
    )
    pygame.draw.rect(
        overlay,
        (*border_color, 190),
        pygame.Rect(0, 0, *overlay_rect.size),
        width=1,
        border_radius=7,
    )
    screen.blit(overlay, overlay_rect.topleft)
    screen.blit(label_surface, (overlay_rect.left + pad_x, overlay_rect.top + pad_y))


def _rounded_game_surface(
    pygame: PygameModule,
    surface: PygameSurface,
    size: tuple[int, int],
    radius: int,
) -> PygameSurface:
    if surface.get_size() != size:
        surface = pygame.transform.smoothscale(surface, size)

    clipped = pygame.Surface(size, pygame.SRCALPHA)
    clipped.blit(surface, (0, 0))
    clipped.blit(
        _rounded_alpha_mask(pygame, size, radius),
        (0, 0),
        special_flags=pygame.BLEND_RGBA_MULT,
    )
    return clipped


def _rounded_alpha_mask(
    pygame: PygameModule,
    size: tuple[int, int],
    radius: int,
) -> PygameSurface:
    cache_key = (*size, radius)
    cached = _GLASS_MASK_CACHE.get(cache_key)
    if cached is not None:
        return cached

    mask = pygame.Surface(size, pygame.SRCALPHA)
    pygame.draw.rect(mask, (255, 255, 255, 255), pygame.Rect(0, 0, *size), border_radius=radius)
    _GLASS_MASK_CACHE[cache_key] = mask
    return mask


def _glass_overlay_surface(
    pygame: PygameModule,
    size: tuple[int, int],
    radius: int,
) -> PygameSurface:
    cache_key = (*size, radius)
    cached = _GLASS_OVERLAY_CACHE.get(cache_key)
    if cached is not None:
        return cached

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


def _draw_glass_vignette(
    *,
    pygame: PygameModule,
    overlay: PygameSurface,
    width: int,
    height: int,
    radius: int,
) -> None:
    max_inset = min(28, width // 9, height // 9)
    for inset in range(max_inset):
        alpha = round(30 * ((max_inset - inset) / max_inset) ** 1.8)
        rect = pygame.Rect(inset, inset, width - (2 * inset), height - (2 * inset))
        pygame.draw.rect(overlay, (0, 0, 0, alpha), rect, width=1, border_radius=radius)
