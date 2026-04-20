# src/rl_fzerox/ui/watch/view/components/cockpit/axis_effects.py
from __future__ import annotations

from rl_fzerox.ui.watch.view.components.cockpit.style import STEER_GAUGE_STYLE
from rl_fzerox.ui.watch.view.components.effects import draw_alpha_polygon
from rl_fzerox.ui.watch.view.screen.theme import PALETTE, Color
from rl_fzerox.ui.watch.view.screen.types import PygameModule, PygameRect, PygameSurface


def draw_steer_glass(
    *,
    pygame: PygameModule,
    screen: PygameSurface,
    rail: PygameRect,
    scale: int,
) -> None:
    style = STEER_GAUGE_STYLE
    tube = rail.inflate(4 * scale, 6 * scale)
    draw_alpha_rounded_rect(
        pygame=pygame,
        screen=screen,
        rect=tube,
        color=style.glass_tint,
        border_radius=tube.height // 2,
    )
    draw_alpha_rounded_rect(
        pygame=pygame,
        screen=screen,
        rect=pygame.Rect(
            tube.left + (5 * scale),
            tube.top + (2 * scale),
            max(1, tube.width - (10 * scale)),
            max(1, tube.height // 3),
        ),
        color=style.glass_highlight,
        border_radius=max(1, tube.height // 5),
    )
    draw_alpha_rounded_rect(
        pygame=pygame,
        screen=screen,
        rect=pygame.Rect(
            tube.left + (6 * scale),
            tube.centery,
            max(1, tube.width - (12 * scale)),
            max(1, (tube.height // 2) - (2 * scale)),
        ),
        color=style.glass_shadow,
        border_radius=max(1, tube.height // 5),
    )
    pygame.draw.line(
        screen,
        style.glass_edge,
        (tube.left + (7 * scale), tube.top + (3 * scale)),
        (tube.right - (7 * scale), tube.top + (3 * scale)),
        width=scale,
    )


def draw_pitch_glass(
    *,
    pygame: PygameModule,
    screen: PygameSurface,
    rail: PygameRect,
    scale: int,
) -> None:
    style = STEER_GAUGE_STYLE
    tube = rail.inflate(6 * scale, 4 * scale)
    draw_alpha_rounded_rect(
        pygame=pygame,
        screen=screen,
        rect=tube,
        color=style.glass_tint,
        border_radius=max(1, tube.width // 2),
    )
    draw_alpha_rounded_rect(
        pygame=pygame,
        screen=screen,
        rect=pygame.Rect(
            tube.left + (2 * scale),
            tube.top + (5 * scale),
            max(1, tube.width // 3),
            max(1, tube.height - (10 * scale)),
        ),
        color=style.glass_highlight,
        border_radius=max(1, tube.width // 5),
    )
    draw_alpha_rounded_rect(
        pygame=pygame,
        screen=screen,
        rect=pygame.Rect(
            tube.centerx,
            tube.top + (6 * scale),
            max(1, (tube.width // 2) - (2 * scale)),
            max(1, tube.height - (12 * scale)),
        ),
        color=style.glass_shadow,
        border_radius=max(1, tube.width // 5),
    )
    pygame.draw.line(
        screen,
        style.glass_edge,
        (tube.left + (3 * scale), tube.top + (7 * scale)),
        (tube.left + (3 * scale), tube.bottom - (7 * scale)),
        width=scale,
    )


def draw_steer_pointer(
    *,
    pygame: PygameModule,
    screen: PygameSurface,
    color: Color,
    x: int,
    rail: PygameRect,
    marker_radius: int,
    scale: int,
) -> None:
    half_width = max(3 * scale, marker_radius - (3 * scale))
    blade = (
        (x, rail.top - (5 * scale)),
        (x + half_width, rail.top + (2 * scale)),
        (x + max(2 * scale, half_width - scale), rail.bottom - (2 * scale)),
        (x, rail.bottom + (5 * scale)),
        (x - max(2 * scale, half_width - scale), rail.bottom - (2 * scale)),
        (x - half_width, rail.top + (2 * scale)),
    )
    pygame.draw.polygon(screen, PALETTE.control_knob_outline, blade)
    inner = (
        (x, rail.top - (3 * scale)),
        (x + max(2 * scale, half_width - scale), rail.top + (3 * scale)),
        (x + max(scale, half_width - (2 * scale)), rail.bottom - (3 * scale)),
        (x, rail.bottom + (3 * scale)),
        (x - max(scale, half_width - (2 * scale)), rail.bottom - (3 * scale)),
        (x - max(2 * scale, half_width - scale), rail.top + (3 * scale)),
    )
    pygame.draw.polygon(screen, color, inner)
    draw_alpha_polygon(
        pygame=pygame,
        screen=screen,
        points=(
            (x - scale, rail.top - scale),
            (x + (2 * scale), rail.top + (3 * scale)),
            (x + scale, rail.bottom - scale),
            (x - scale, rail.bottom + scale),
        ),
        color=STEER_GAUGE_STYLE.pointer_glint,
    )


def draw_pitch_pointer(
    *,
    pygame: PygameModule,
    screen: PygameSurface,
    color: Color,
    y: int,
    rail: PygameRect,
    marker_radius: int,
    scale: int,
) -> None:
    half_height = max(3 * scale, marker_radius - (3 * scale))
    blade = (
        (rail.left - (5 * scale), y),
        (rail.left + (2 * scale), y - half_height),
        (rail.right - (2 * scale), y - max(2 * scale, half_height - scale)),
        (rail.right + (5 * scale), y),
        (rail.right - (2 * scale), y + max(2 * scale, half_height - scale)),
        (rail.left + (2 * scale), y + half_height),
    )
    pygame.draw.polygon(screen, PALETTE.control_knob_outline, blade)
    inner = (
        (rail.left - (3 * scale), y),
        (rail.left + (3 * scale), y - max(2 * scale, half_height - scale)),
        (rail.right - (3 * scale), y - max(scale, half_height - (2 * scale))),
        (rail.right + (3 * scale), y),
        (rail.right - (3 * scale), y + max(scale, half_height - (2 * scale))),
        (rail.left + (3 * scale), y + max(2 * scale, half_height - scale)),
    )
    pygame.draw.polygon(screen, color, inner)
    draw_alpha_polygon(
        pygame=pygame,
        screen=screen,
        points=(
            (rail.left - scale, y - scale),
            (rail.left + (3 * scale), y - (2 * scale)),
            (rail.right - scale, y - scale),
            (rail.right + scale, y + scale),
        ),
        color=STEER_GAUGE_STYLE.pointer_glint,
    )


def draw_alpha_rounded_rect(
    *,
    pygame: PygameModule,
    screen: PygameSurface,
    rect: PygameRect,
    color: tuple[int, int, int, int],
    border_radius: int,
) -> None:
    surface = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
    local_rect = pygame.Rect(0, 0, rect.width, rect.height)
    pygame.draw.rect(surface, color, local_rect, border_radius=border_radius)
    screen.blit(surface, rect.topleft)
