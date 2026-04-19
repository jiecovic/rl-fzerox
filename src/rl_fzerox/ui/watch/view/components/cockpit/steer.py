# src/rl_fzerox/ui/watch/view/components/cockpit/steer.py
from __future__ import annotations

from rl_fzerox.ui.watch.view.components.cockpit.style import (
    STEER_AXIS_GUIDE,
    STEER_GAUGE_STYLE,
    THRUST_COLUMN_STYLE,
)
from rl_fzerox.ui.watch.view.components.effects import (
    draw_alpha_polygon as _draw_alpha_polygon,
)
from rl_fzerox.ui.watch.view.screen.theme import PALETTE, Color


def _draw_steer_instrument(
    *,
    pygame,
    screen,
    track,
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


def _draw_pitch_instrument(
    *,
    pygame,
    screen,
    track,
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
    pygame,
    screen,
    track,
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
    pygame.draw.rect(screen, style.rail_fill, rail, border_radius=style.rail_radius * scale)
    _draw_steer_axis_guides(pygame=pygame, screen=screen, rail=rail, scale=scale)
    _draw_steer_fill(pygame=pygame, screen=screen, rail=rail, value=value, scale=scale)
    _draw_steer_glass(pygame=pygame, screen=screen, rail=rail, scale=scale)
    pygame.draw.rect(
        screen,
        PALETTE.control_lever_border,
        rail,
        width=THRUST_COLUMN_STYLE.border_width * scale,
        border_radius=style.rail_radius * scale,
    )
    _draw_steer_pointer(
        pygame=pygame,
        screen=screen,
        color=_steer_indicator_color(value),
        x=_steer_track_x(track=rail, value=value),
        rail=rail,
        marker_radius=marker_radius,
        scale=scale,
    )


def _draw_pitch_instrument_surface(
    *,
    pygame,
    screen,
    track,
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
    pygame.draw.rect(screen, style.rail_fill, rail, border_radius=style.rail_radius * scale)
    _draw_pitch_axis_guides(pygame=pygame, screen=screen, rail=rail, scale=scale)
    _draw_pitch_fill(pygame=pygame, screen=screen, rail=rail, value=value, scale=scale)
    _draw_pitch_glass(pygame=pygame, screen=screen, rail=rail, scale=scale)
    pygame.draw.rect(
        screen,
        PALETTE.control_lever_border,
        rail,
        width=THRUST_COLUMN_STYLE.border_width * scale,
        border_radius=style.rail_radius * scale,
    )
    _draw_pitch_pointer(
        pygame=pygame,
        screen=screen,
        color=_steer_indicator_color(value),
        y=_pitch_track_y(track=rail, value=value),
        rail=rail,
        marker_radius=marker_radius,
        scale=scale,
    )


def _draw_steer_fill(*, pygame, screen, rail, value: float, scale: int) -> None:
    value = max(-1.0, min(1.0, value))
    magnitude = abs(value)
    if magnitude == 0.0:
        return

    fill_color = _steer_indicator_color(value)
    _draw_steer_segment(
        pygame=pygame,
        screen=screen,
        track=rail.inflate(0, -STEER_GAUGE_STYLE.fill_vertical_shrink * scale),
        start=0.0,
        end=value,
        color=fill_color,
    )


def _draw_pitch_fill(*, pygame, screen, rail, value: float, scale: int) -> None:
    value = max(-1.0, min(1.0, value))
    if value == 0.0:
        return

    _draw_pitch_segment(
        pygame=pygame,
        screen=screen,
        track=rail.inflate(-STEER_GAUGE_STYLE.fill_vertical_shrink * scale, 0),
        start=0.0,
        end=value,
        color=_steer_indicator_color(value),
    )


def _draw_pitch_segment(*, pygame, screen, track, start: float, end: float, color) -> None:
    start_y = _pitch_track_y(track=track, value=start)
    end_y = _pitch_track_y(track=track, value=end)
    segment_y = min(start_y, end_y)
    segment_height = max(1, abs(end_y - start_y))
    pygame.draw.rect(screen, color, pygame.Rect(track.x, segment_y, track.width, segment_height))


def _draw_steer_segment(*, pygame, screen, track, start: float, end: float, color) -> None:
    start_x = _steer_track_x(track=track, value=start)
    end_x = _steer_track_x(track=track, value=end)
    segment_x = min(start_x, end_x)
    segment_width = max(1, abs(end_x - start_x))
    segment_rect = pygame.Rect(segment_x, track.y, segment_width, track.height)
    pygame.draw.rect(screen, color, segment_rect)


def _steer_indicator_color(value: float) -> Color:
    magnitude = abs(max(-1.0, min(1.0, value)))
    if magnitude <= STEER_AXIS_GUIDE.deadzone:
        return STEER_GAUGE_STYLE.deadzone_color
    if magnitude >= STEER_AXIS_GUIDE.saturation:
        return STEER_GAUGE_STYLE.extreme_color
    return PALETTE.text_primary


def _draw_steer_axis_guides(*, pygame, screen, rail, scale: int) -> None:
    style = STEER_GAUGE_STYLE
    center_x = _steer_track_x(track=rail, value=0.0)
    deadzone_left = _steer_track_x(track=rail, value=-STEER_AXIS_GUIDE.deadzone)
    deadzone_right = _steer_track_x(track=rail, value=STEER_AXIS_GUIDE.deadzone)
    visual_left, visual_right = _visible_deadzone_span(
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
        marker_x = _steer_track_x(track=rail, value=value)
        pygame.draw.line(
            screen,
            style.deadzone_line,
            (marker_x, rail.top - (style.deadzone_tick_extension * scale)),
            (marker_x, rail.bottom + (style.deadzone_tick_extension * scale)),
            width=style.deadzone_tick_width * scale,
        )
    for value in (-STEER_AXIS_GUIDE.saturation, STEER_AXIS_GUIDE.saturation):
        marker_x = _steer_track_x(track=rail, value=value)
        pygame.draw.line(
            screen,
            style.extreme_color,
            (marker_x, rail.top),
            (marker_x, rail.bottom),
            width=scale,
        )


def _draw_pitch_axis_guides(*, pygame, screen, rail, scale: int) -> None:
    style = STEER_GAUGE_STYLE
    center_y = _pitch_track_y(track=rail, value=0.0)
    deadzone_top = _pitch_track_y(track=rail, value=STEER_AXIS_GUIDE.deadzone)
    deadzone_bottom = _pitch_track_y(track=rail, value=-STEER_AXIS_GUIDE.deadzone)
    visual_top, visual_bottom = _visible_deadzone_span(
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
        marker_y = _pitch_track_y(track=rail, value=value)
        pygame.draw.line(
            screen,
            style.deadzone_line,
            (rail.left - (style.deadzone_tick_extension * scale), marker_y),
            (rail.right + (style.deadzone_tick_extension * scale), marker_y),
            width=style.deadzone_tick_width * scale,
        )
    for value in (-STEER_AXIS_GUIDE.saturation, STEER_AXIS_GUIDE.saturation):
        marker_y = _pitch_track_y(track=rail, value=value)
        pygame.draw.line(
            screen,
            style.extreme_color,
            (rail.left, marker_y),
            (rail.right, marker_y),
            width=scale,
        )


def _draw_steer_glass(*, pygame, screen, rail, scale: int) -> None:
    style = STEER_GAUGE_STYLE
    tube = rail.inflate(4 * scale, 6 * scale)
    _draw_alpha_rounded_rect(
        pygame=pygame,
        screen=screen,
        rect=tube,
        color=style.glass_tint,
        border_radius=tube.height // 2,
    )
    _draw_alpha_rounded_rect(
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
    _draw_alpha_rounded_rect(
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


def _draw_pitch_glass(*, pygame, screen, rail, scale: int) -> None:
    style = STEER_GAUGE_STYLE
    tube = rail.inflate(6 * scale, 4 * scale)
    _draw_alpha_rounded_rect(
        pygame=pygame,
        screen=screen,
        rect=tube,
        color=style.glass_tint,
        border_radius=max(1, tube.width // 2),
    )
    _draw_alpha_rounded_rect(
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
    _draw_alpha_rounded_rect(
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


def _draw_steer_pointer(
    *,
    pygame,
    screen,
    color: Color,
    x: int,
    rail,
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
    _draw_alpha_polygon(
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


def _draw_pitch_pointer(
    *,
    pygame,
    screen,
    color: Color,
    y: int,
    rail,
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
    _draw_alpha_polygon(
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


def _draw_alpha_rounded_rect(
    *,
    pygame,
    screen,
    rect,
    color: tuple[int, int, int, int],
    border_radius: int,
) -> None:
    surface = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
    local_rect = pygame.Rect(0, 0, rect.width, rect.height)
    pygame.draw.rect(surface, color, local_rect, border_radius=border_radius)
    screen.blit(surface, rect.topleft)


def _steer_track_x(*, track, value: float) -> int:
    value = max(-1.0, min(1.0, value))
    return track.centerx + round((track.width // 2) * value)


def _pitch_track_y(*, track, value: float) -> int:
    value = max(-1.0, min(1.0, value))
    return track.centery - round((track.height // 2) * value)


def _visible_deadzone_span(
    *,
    center: int,
    lower: int,
    upper: int,
    minimum: int,
    min_bound: int,
    max_bound: int,
) -> tuple[int, int]:
    if upper - lower >= minimum:
        return lower, upper

    half_width = minimum // 2
    lower = max(min_bound, center - half_width)
    upper = min(max_bound, center + half_width)
    return lower, upper
