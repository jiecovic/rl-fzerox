# src/rl_fzerox/ui/watch/view/components/effects.py
from __future__ import annotations

from rl_fzerox.ui.watch.view.screen.theme import Color

GLASS_HIGHLIGHT: Color = (170, 190, 210)
GLASS_LOWLIGHT: Color = (8, 11, 15)
GLASS_SHEEN: tuple[int, int, int, int] = (238, 248, 255, 82)
GLASS_SHADOW: tuple[int, int, int, int] = (0, 0, 0, 44)
GLASS_EDGE_GLOW: tuple[int, int, int, int] = (126, 214, 170, 48)
BOOST_EDGE_GLOW: tuple[int, int, int, int] = (126, 214, 170, 52)


def blend_color(start: Color, end: Color, weight: float) -> Color:
    weight = max(0.0, min(1.0, weight))
    return (
        round(start[0] + ((end[0] - start[0]) * weight)),
        round(start[1] + ((end[1] - start[1]) * weight)),
        round(start[2] + ((end[2] - start[2]) * weight)),
    )


def offset_points(
    points: tuple[tuple[int, int], ...],
    *,
    dx: int,
    dy: int,
) -> tuple[tuple[int, int], ...]:
    return tuple((x + dx, y + dy) for x, y in points)


def draw_alpha_polygon(
    *,
    pygame,
    screen,
    points: tuple[tuple[int, int], ...],
    color: tuple[int, int, int, int],
) -> None:
    min_x = min(x for x, _ in points)
    max_x = max(x for x, _ in points)
    min_y = min(y for _, y in points)
    max_y = max(y for _, y in points)
    surface = pygame.Surface((max_x - min_x + 1, max_y - min_y + 1), pygame.SRCALPHA)
    local_points = tuple((x - min_x, y - min_y) for x, y in points)
    pygame.draw.polygon(surface, color, local_points)
    screen.blit(surface, (min_x, min_y))


def draw_alpha_circle(
    *,
    pygame,
    screen,
    center: tuple[int, int],
    radius: int,
    color: tuple[int, int, int, int],
) -> None:
    size = (radius * 2) + 1
    surface = pygame.Surface((size, size), pygame.SRCALPHA)
    pygame.draw.circle(surface, color, (radius, radius), radius)
    screen.blit(surface, (center[0] - radius, center[1] - radius))


def draw_alpha_rect(*, pygame, screen, rect, color: tuple[int, int, int, int]) -> None:
    surface = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
    surface.fill(color)
    screen.blit(surface, rect.topleft)


def draw_glass_track_overlay(*, pygame, screen, track) -> None:
    draw_alpha_rect(
        pygame=pygame,
        screen=screen,
        rect=pygame.Rect(track.left + 4, track.top + 2, max(1, track.width - 8), 3),
        color=(255, 255, 255, 46),
    )
    draw_alpha_rect(
        pygame=pygame,
        screen=screen,
        rect=pygame.Rect(
            track.left + 4,
            track.top + 3,
            max(1, track.width - 8),
            max(1, track.height // 2),
        ),
        color=(255, 255, 255, 24),
    )
    draw_alpha_rect(
        pygame=pygame,
        screen=screen,
        rect=pygame.Rect(
            track.left + 4,
            track.centery,
            max(1, track.width - 8),
            max(1, (track.height // 2) - 2),
        ),
        color=(0, 0, 0, 26),
    )
    pygame.draw.line(
        screen,
        GLASS_HIGHLIGHT,
        (track.left + 4, track.top + 2),
        (track.right - 4, track.top + 2),
        width=1,
    )
    pygame.draw.line(
        screen,
        GLASS_LOWLIGHT,
        (track.left + 4, track.bottom - 2),
        (track.right - 4, track.bottom - 2),
        width=1,
    )


def draw_glass_column_overlay(*, pygame, screen, track) -> None:
    draw_alpha_rect(
        pygame=pygame,
        screen=screen,
        rect=pygame.Rect(track.left + 3, track.top + 4, 4, max(1, track.height - 8)),
        color=(255, 255, 255, 78),
    )
    draw_alpha_rect(
        pygame=pygame,
        screen=screen,
        rect=pygame.Rect(track.left + 4, track.top + 3, max(1, track.width - 8), track.height // 2),
        color=(255, 255, 255, 38),
    )
    draw_alpha_rect(
        pygame=pygame,
        screen=screen,
        rect=pygame.Rect(track.centerx, track.top + 5, 2, max(1, track.height - 10)),
        color=(255, 255, 255, 34),
    )
    draw_alpha_rect(
        pygame=pygame,
        screen=screen,
        rect=pygame.Rect(
            track.left + 4, track.centery, max(1, track.width - 8), track.height // 2 - 4
        ),
        color=(0, 0, 0, 34),
    )
    pygame.draw.line(
        screen,
        GLASS_LOWLIGHT,
        (track.left + 4, track.bottom - 3),
        (track.right - 4, track.bottom - 3),
        width=1,
    )
