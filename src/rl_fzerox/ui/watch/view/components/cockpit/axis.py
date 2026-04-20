# src/rl_fzerox/ui/watch/view/components/cockpit/axis.py
from __future__ import annotations

from rl_fzerox.ui.watch.view.components.cockpit.axis_effects import (
    draw_pitch_glass,
    draw_pitch_pointer,
    draw_steer_glass,
    draw_steer_pointer,
)
from rl_fzerox.ui.watch.view.components.cockpit.axis_geometry import (
    pitch_track_y,
    steer_indicator_color,
    steer_track_x,
    visible_deadzone_span,
)
from rl_fzerox.ui.watch.view.components.cockpit.style import (
    STEER_AXIS_GUIDE,
    STEER_GAUGE_STYLE,
    THRUST_COLUMN_STYLE,
)
from rl_fzerox.ui.watch.view.screen.theme import PALETTE, Color
from rl_fzerox.ui.watch.view.screen.types import PygameModule, PygameRect, PygameSurface


def draw_steer_instrument(
    *,
    pygame: PygameModule,
    screen: PygameSurface,
    track: PygameRect,
    value: float,
    marker_radius: int,
) -> None:
    style = STEER_GAUGE_STYLE
    target = track.inflate(*style.target_inflate)
    scale = style.render_scale
    surface = pygame.Surface((target.width * scale, target.height * scale), pygame.SRCALPHA)
    local_track = pygame.Rect(
        (track.left - target.left) * scale,
        (track.top - target.top) * scale,
        track.width * scale,
        track.height * scale,
    )
    _draw_steer_instrument_surface(
        pygame=pygame,
        screen=surface,
        track=local_track,
        value=value,
        marker_radius=marker_radius * scale,
        scale=scale,
    )
    rendered = pygame.transform.smoothscale(surface, target.size)
    screen.blit(rendered, target.topleft)


def draw_pitch_instrument(
    *,
    pygame: PygameModule,
    screen: PygameSurface,
    track: PygameRect,
    value: float,
    marker_radius: int,
) -> None:
    style = STEER_GAUGE_STYLE
    target = track.inflate(style.target_inflate[1], style.target_inflate[0])
    scale = style.render_scale
    surface = pygame.Surface((target.width * scale, target.height * scale), pygame.SRCALPHA)
    local_track = pygame.Rect(
        (track.left - target.left) * scale,
        (track.top - target.top) * scale,
        track.width * scale,
        track.height * scale,
    )
    _draw_pitch_instrument_surface(
        pygame=pygame,
        screen=surface,
        track=local_track,
        value=value,
        marker_radius=marker_radius * scale,
        scale=scale,
    )
    rendered = pygame.transform.smoothscale(surface, target.size)
    screen.blit(rendered, target.topleft)


def _draw_steer_instrument_surface(
    *,
    pygame: PygameModule,
    screen: PygameSurface,
    track: PygameRect,
    value: float,
    marker_radius: int,
    scale: int,
) -> None:
    style = STEER_GAUGE_STYLE
    bezel = track.inflate(style.bezel_inflate[0] * scale, style.bezel_inflate[1] * scale)
    rail = pygame.Rect(
        track.left + (style.rail_inset[0] * scale),
        track.top + (style.rail_inset[1] * scale),
        max(1, track.width - (2 * style.rail_inset[0] * scale)),
        max(1, track.height - (style.rail_vertical_shrink * scale)),
    )
    _draw_bezel(pygame=pygame, screen=screen, bezel=bezel, scale=scale)
    pygame.draw.rect(screen, style.rail_fill, rail, border_radius=style.rail_radius * scale)
    _draw_steer_axis_guides(pygame=pygame, screen=screen, rail=rail, scale=scale)
    _draw_steer_fill(pygame=pygame, screen=screen, rail=rail, value=value, scale=scale)
    draw_steer_glass(pygame=pygame, screen=screen, rail=rail, scale=scale)
    _draw_rail_outline(pygame=pygame, screen=screen, rail=rail, scale=scale)
    draw_steer_pointer(
        pygame=pygame,
        screen=screen,
        color=steer_indicator_color(value),
        x=steer_track_x(track=rail, value=value),
        rail=rail,
        marker_radius=marker_radius,
        scale=scale,
    )


def _draw_pitch_instrument_surface(
    *,
    pygame: PygameModule,
    screen: PygameSurface,
    track: PygameRect,
    value: float,
    marker_radius: int,
    scale: int,
) -> None:
    style = STEER_GAUGE_STYLE
    bezel = track.inflate(style.bezel_inflate[1] * scale, style.bezel_inflate[0] * scale)
    rail = pygame.Rect(
        track.left + (style.rail_inset[1] * scale),
        track.top + (style.rail_inset[0] * scale),
        max(1, track.width - (2 * style.rail_inset[1] * scale)),
        max(1, track.height - (2 * style.rail_inset[0] * scale)),
    )
    _draw_bezel(pygame=pygame, screen=screen, bezel=bezel, scale=scale)
    pygame.draw.rect(screen, style.rail_fill, rail, border_radius=style.rail_radius * scale)
    _draw_pitch_axis_guides(pygame=pygame, screen=screen, rail=rail, scale=scale)
    _draw_pitch_fill(pygame=pygame, screen=screen, rail=rail, value=value, scale=scale)
    draw_pitch_glass(pygame=pygame, screen=screen, rail=rail, scale=scale)
    _draw_rail_outline(pygame=pygame, screen=screen, rail=rail, scale=scale)
    draw_pitch_pointer(
        pygame=pygame,
        screen=screen,
        color=steer_indicator_color(value),
        y=pitch_track_y(track=rail, value=value),
        rail=rail,
        marker_radius=marker_radius,
        scale=scale,
    )


def _draw_bezel(
    *, pygame: PygameModule, screen: PygameSurface, bezel: PygameRect, scale: int
) -> None:
    style = STEER_GAUGE_STYLE
    pygame.draw.rect(
        screen,
        style.shadow_fill,
        bezel.move(style.shadow_offset[0] * scale, style.shadow_offset[1] * scale),
        border_radius=style.bezel_radius * scale,
    )
    pygame.draw.rect(
        screen,
        style.bezel_fill,
        bezel,
        border_radius=style.bezel_radius * scale,
    )
    pygame.draw.rect(
        screen,
        style.bezel_border,
        bezel,
        width=THRUST_COLUMN_STYLE.border_width * scale,
        border_radius=style.bezel_radius * scale,
    )


def _draw_rail_outline(
    *, pygame: PygameModule, screen: PygameSurface, rail: PygameRect, scale: int
) -> None:
    pygame.draw.rect(
        screen,
        PALETTE.control_lever_border,
        rail,
        width=THRUST_COLUMN_STYLE.border_width * scale,
        border_radius=STEER_GAUGE_STYLE.rail_radius * scale,
    )


def _draw_steer_fill(
    *,
    pygame: PygameModule,
    screen: PygameSurface,
    rail: PygameRect,
    value: float,
    scale: int,
) -> None:
    value = max(-1.0, min(1.0, value))
    if abs(value) == 0.0:
        return

    _draw_steer_segment(
        pygame=pygame,
        screen=screen,
        track=rail.inflate(0, -STEER_GAUGE_STYLE.fill_vertical_shrink * scale),
        start=0.0,
        end=value,
        color=steer_indicator_color(value),
    )


def _draw_pitch_fill(
    *,
    pygame: PygameModule,
    screen: PygameSurface,
    rail: PygameRect,
    value: float,
    scale: int,
) -> None:
    value = max(-1.0, min(1.0, value))
    if value == 0.0:
        return

    _draw_pitch_segment(
        pygame=pygame,
        screen=screen,
        track=rail.inflate(-STEER_GAUGE_STYLE.fill_vertical_shrink * scale, 0),
        start=0.0,
        end=value,
        color=steer_indicator_color(value),
    )


def _draw_pitch_segment(
    *,
    pygame: PygameModule,
    screen: PygameSurface,
    track: PygameRect,
    start: float,
    end: float,
    color: Color,
) -> None:
    start_y = pitch_track_y(track=track, value=start)
    end_y = pitch_track_y(track=track, value=end)
    segment_y = min(start_y, end_y)
    segment_height = max(1, abs(end_y - start_y))
    pygame.draw.rect(screen, color, pygame.Rect(track.x, segment_y, track.width, segment_height))


def _draw_steer_segment(
    *,
    pygame: PygameModule,
    screen: PygameSurface,
    track: PygameRect,
    start: float,
    end: float,
    color: Color,
) -> None:
    start_x = steer_track_x(track=track, value=start)
    end_x = steer_track_x(track=track, value=end)
    segment_x = min(start_x, end_x)
    segment_width = max(1, abs(end_x - start_x))
    segment_rect = pygame.Rect(segment_x, track.y, segment_width, track.height)
    pygame.draw.rect(screen, color, segment_rect)


def _draw_steer_axis_guides(
    *, pygame: PygameModule, screen: PygameSurface, rail: PygameRect, scale: int
) -> None:
    style = STEER_GAUGE_STYLE
    center_x = steer_track_x(track=rail, value=0.0)
    deadzone_left = steer_track_x(track=rail, value=-STEER_AXIS_GUIDE.deadzone)
    deadzone_right = steer_track_x(track=rail, value=STEER_AXIS_GUIDE.deadzone)
    visual_left, visual_right = visible_deadzone_span(
        center=center_x,
        lower=deadzone_left,
        upper=deadzone_right,
        minimum=style.deadzone_visual_min_size * scale,
        min_bound=rail.left,
        max_bound=rail.right,
    )
    pygame.draw.rect(
        screen,
        style.deadzone_color,
        pygame.Rect(
            visual_left,
            rail.top + (2 * scale),
            max(1, visual_right - visual_left),
            max(1, rail.height - (4 * scale)),
        ),
    )
    pygame.draw.line(
        screen,
        STEER_GAUGE_STYLE.center_line,
        (center_x, rail.top),
        (center_x, rail.bottom),
        width=scale,
    )
    for value in (-STEER_AXIS_GUIDE.deadzone, STEER_AXIS_GUIDE.deadzone):
        marker_x = steer_track_x(track=rail, value=value)
        pygame.draw.line(
            screen,
            style.deadzone_line,
            (marker_x, rail.top - (style.deadzone_tick_extension * scale)),
            (marker_x, rail.bottom + (style.deadzone_tick_extension * scale)),
            width=style.deadzone_tick_width * scale,
        )
    for value in (-STEER_AXIS_GUIDE.saturation, STEER_AXIS_GUIDE.saturation):
        marker_x = steer_track_x(track=rail, value=value)
        pygame.draw.line(
            screen,
            style.extreme_color,
            (marker_x, rail.top),
            (marker_x, rail.bottom),
            width=scale,
        )


def _draw_pitch_axis_guides(
    *, pygame: PygameModule, screen: PygameSurface, rail: PygameRect, scale: int
) -> None:
    style = STEER_GAUGE_STYLE
    center_y = pitch_track_y(track=rail, value=0.0)
    deadzone_top = pitch_track_y(track=rail, value=STEER_AXIS_GUIDE.deadzone)
    deadzone_bottom = pitch_track_y(track=rail, value=-STEER_AXIS_GUIDE.deadzone)
    visual_top, visual_bottom = visible_deadzone_span(
        center=center_y,
        lower=deadzone_top,
        upper=deadzone_bottom,
        minimum=style.deadzone_visual_min_size * scale,
        min_bound=rail.top,
        max_bound=rail.bottom,
    )
    pygame.draw.rect(
        screen,
        style.deadzone_color,
        pygame.Rect(
            rail.left + (2 * scale),
            visual_top,
            max(1, rail.width - (4 * scale)),
            max(1, visual_bottom - visual_top),
        ),
    )
    pygame.draw.line(
        screen,
        STEER_GAUGE_STYLE.center_line,
        (rail.left, center_y),
        (rail.right, center_y),
        width=scale,
    )
    for value in (-STEER_AXIS_GUIDE.deadzone, STEER_AXIS_GUIDE.deadzone):
        marker_y = pitch_track_y(track=rail, value=value)
        pygame.draw.line(
            screen,
            style.deadzone_line,
            (rail.left - (style.deadzone_tick_extension * scale), marker_y),
            (rail.right + (style.deadzone_tick_extension * scale), marker_y),
            width=style.deadzone_tick_width * scale,
        )
    for value in (-STEER_AXIS_GUIDE.saturation, STEER_AXIS_GUIDE.saturation):
        marker_y = pitch_track_y(track=rail, value=value)
        pygame.draw.line(
            screen,
            style.extreme_color,
            (rail.left, marker_y),
            (rail.right, marker_y),
            width=scale,
        )
