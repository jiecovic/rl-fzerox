# src/rl_fzerox/ui/watch/view/components/cockpit/thrust.py
from __future__ import annotations

from rl_fzerox.ui.watch.view.components.cockpit.style import (
    BUTTON_FACE_STYLE,
    THRUST_COLUMN_STYLE,
)
from rl_fzerox.ui.watch.view.components.effects import (
    draw_glass_column_overlay as _draw_glass_column_overlay,
)
from rl_fzerox.ui.watch.view.screen.theme import PALETTE, Color
from rl_fzerox.ui.watch.view.screen.types import PygameModule, PygameRect, PygameSurface


def _draw_thrust_column(
    *,
    pygame: PygameModule,
    screen: PygameSurface,
    x: int,
    y: int,
    level: float,
    deadzone_threshold: float | None,
    full_threshold: float | None,
    fill_color: Color,
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
        BUTTON_FACE_STYLE.shadow,
        track.move(1, 2),
        border_radius=track.width // 2,
    )
    pygame.draw.rect(
        screen,
        PALETTE.control_track,
        track,
        border_radius=track.width // 2,
    )

    style = THRUST_COLUMN_STYLE
    tall_meter = height >= 96
    segment_count = style.tall_segment_count if tall_meter else style.compact_segment_count
    segment_gap = style.tall_segment_gap if tall_meter else style.compact_segment_gap
    segment_heights = _segment_heights(segment_count=segment_count, tall_meter=tall_meter)
    stack_height = sum(segment_heights) + ((segment_count - 1) * segment_gap)
    stack_top = track.top + ((track.height - stack_height) // 2)
    lit_segments = round(level * segment_count)
    segment_y = stack_top
    for segment_index in range(segment_count):
        height_index = segment_count - 1 - segment_index
        segment_height = segment_heights[height_index]
        segment = pygame.Rect(
            track.x + style.segment_horizontal_inset,
            segment_y,
            track.width - (2 * style.segment_horizontal_inset),
            segment_height,
        )
        if segment_index >= segment_count - lit_segments:
            pygame.draw.rect(screen, fill_color, segment, border_radius=1)
        else:
            pygame.draw.rect(
                screen,
                THRUST_COLUMN_STYLE.unlit_segment_fill,
                segment,
                border_radius=1,
            )
        segment_y += segment_height + segment_gap

    _draw_glass_column_overlay(pygame=pygame, screen=screen, track=track)

    _draw_zone_marker(
        pygame=pygame,
        screen=screen,
        track=track,
        threshold=deadzone_threshold,
        color=style.deadzone_marker,
    )
    _draw_zone_marker(
        pygame=pygame,
        screen=screen,
        track=track,
        threshold=full_threshold,
        color=style.full_zone_marker,
    )

    pygame.draw.rect(
        screen,
        PALETTE.control_lever_border,
        track,
        width=THRUST_COLUMN_STYLE.border_width,
        border_radius=3,
    )


def _draw_zone_marker(
    *,
    pygame: PygameModule,
    screen: PygameSurface,
    track: PygameRect,
    threshold: float | None,
    color: Color,
) -> None:
    if threshold is None or threshold <= 0.0 or threshold >= 1.0:
        return
    style = THRUST_COLUMN_STYLE
    threshold = max(0.0, min(1.0, threshold))
    threshold_y = track.bottom - round(track.height * threshold)
    pygame.draw.line(
        screen,
        color,
        (track.left - style.marker_extension, threshold_y),
        (track.right + style.marker_extension, threshold_y),
        width=style.marker_width,
    )


def _segment_heights(*, segment_count: int, tall_meter: bool) -> tuple[int, ...]:
    style = THRUST_COLUMN_STYLE
    if not tall_meter:
        return (style.compact_segment_height,) * segment_count
    heights = [style.tall_segment_height] * segment_count
    for index in (0, segment_count // 2, segment_count - 1):
        heights[index] = style.tall_segment_trimmed_height
    return tuple(heights)
