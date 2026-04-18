# src/rl_fzerox/ui/watch/view/components/cockpit/thrust.py
from __future__ import annotations

from rl_fzerox.ui.watch.view.components.cockpit.style import (
    BUTTON_SHADOW,
    COCKPIT_GRID,
    THRUST_COLUMN_BORDER_WIDTH,
    THRUST_WARNING_FILL,
)
from rl_fzerox.ui.watch.view.components.effects import (
    draw_glass_column_overlay as _draw_glass_column_overlay,
)
from rl_fzerox.ui.watch.view.screen.theme import PALETTE


def _draw_thrust_column(
    *,
    pygame,
    screen,
    x: int,
    y: int,
    level: float,
    threshold: float | None,
    fill_color,
    width: int,
    height: int,
) -> None:
    level = max(0.0, min(1.0, level))
    track = pygame.Rect(
        x,
        y,
        width,
        height,
    )
    pygame.draw.rect(
        screen,
        BUTTON_SHADOW,
        track.move(1, 2),
        border_radius=track.width // 2,
    )
    pygame.draw.rect(
        screen,
        PALETTE.control_track,
        track,
        border_radius=track.width // 2,
    )

    segment_count = 9 if height >= 96 else 7
    segment_gap = 2 if height >= 96 else 1
    segment_height = 9 if height >= 96 else 8
    stack_height = (segment_count * segment_height) + ((segment_count - 1) * segment_gap)
    stack_top = track.top + ((track.height - stack_height) // 2)
    lit_segments = round(level * segment_count)
    for segment_index in range(segment_count):
        segment_y = stack_top
        segment_y += (segment_count - 1 - segment_index) * (segment_height + segment_gap)
        segment = pygame.Rect(track.x + 4, segment_y, track.width - 8, segment_height)
        if segment_index < lit_segments:
            pygame.draw.rect(screen, fill_color, segment, border_radius=1)
            continue
        pygame.draw.rect(screen, COCKPIT_GRID, segment, border_radius=1)

    _draw_glass_column_overlay(pygame=pygame, screen=screen, track=track)

    if threshold is not None:
        threshold = max(0.0, min(1.0, threshold))
        threshold_y = track.bottom - round(track.height * threshold)
        pygame.draw.line(
            screen,
            THRUST_WARNING_FILL,
            (track.left - 5, threshold_y),
            (track.right + 5, threshold_y),
            width=2,
        )

    pygame.draw.rect(
        screen,
        PALETTE.control_lever_border,
        track,
        width=THRUST_COLUMN_BORDER_WIDTH,
        border_radius=3,
    )
