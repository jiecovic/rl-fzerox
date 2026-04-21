# src/rl_fzerox/ui/watch/view/components/cockpit/speed.py
from __future__ import annotations

import math

from rl_fzerox.ui.watch.view.components.cockpit.style import SPEED_GAUGE_STYLE
from rl_fzerox.ui.watch.view.screen.theme import PALETTE, Color
from rl_fzerox.ui.watch.view.screen.types import (
    PygameModule,
    PygameRect,
    PygameSurface,
    RenderFont,
)


def draw_speed_gauge(
    *,
    pygame: PygameModule,
    screen: PygameSurface,
    rect: PygameRect,
    speed_kph: float | None,
    energy_fraction: float | None,
    value_font: RenderFont,
) -> None:
    style = SPEED_GAUGE_STYLE
    pygame.draw.rect(screen, style.fill, rect, border_radius=4)
    inner_rect = rect.inflate(-6, -6)
    pygame.draw.rect(screen, style.inner_fill, inner_rect, border_radius=3)
    for line_y in range(inner_rect.top + 4, inner_rect.bottom - 3, 5):
        pygame.draw.line(
            screen,
            style.scanline,
            (inner_rect.left + 3, line_y),
            (inner_rect.right - 4, line_y),
        )
    pygame.draw.rect(screen, style.border, rect, width=2, border_radius=4)
    pygame.draw.rect(screen, style.inner_border, inner_rect, width=1, border_radius=3)

    scale = style.render_scale
    overlay = pygame.Surface((rect.width * scale, rect.height * scale), pygame.SRCALPHA)
    local_rect = pygame.Rect(0, 0, rect.width * scale, rect.height * scale)
    center = (
        local_rect.centerx,
        local_rect.bottom - (style.center_bottom_padding * scale),
    )
    radius = max(
        8 * scale,
        min(
            (local_rect.width // 2) - (style.radius_padding * scale),
            local_rect.height - (style.vertical_radius_padding * scale),
        ),
    )
    normalized = _normalized_speed(speed_kph)

    _draw_arc(
        pygame=pygame,
        surface=overlay,
        center=center,
        radius=radius,
        start=0.0,
        end=1.0,
        color=style.arc_idle,
        width=style.arc_width * scale,
    )
    _draw_arc(
        pygame=pygame,
        surface=overlay,
        center=center,
        radius=radius,
        start=style.red_zone_start,
        end=1.0,
        color=style.red_zone,
        width=style.arc_width * scale,
    )
    _draw_arc(
        pygame=pygame,
        surface=overlay,
        center=center,
        radius=radius,
        start=0.0,
        end=normalized,
        color=_speed_color(normalized),
        width=style.active_arc_width * scale,
    )
    _draw_ticks(pygame=pygame, surface=overlay, center=center, radius=radius, scale=scale)
    _draw_needle(
        pygame=pygame,
        surface=overlay,
        center=center,
        radius=radius,
        normalized=normalized,
        scale=scale,
    )
    screen.blit(pygame.transform.smoothscale(overlay, rect.size), rect.topleft)
    _draw_energy_meter(
        pygame=pygame,
        screen=screen,
        rect=rect,
        energy_fraction=energy_fraction,
    )
    _draw_speed_text(
        screen=screen,
        rect=rect,
        speed_kph=speed_kph,
        value_font=value_font,
    )


def _normalized_speed(speed_kph: float | None) -> float:
    if speed_kph is None or not math.isfinite(speed_kph):
        return 0.0
    return max(0.0, min(1.0, speed_kph / SPEED_GAUGE_STYLE.max_kph))


def _draw_arc(
    *,
    pygame: PygameModule,
    surface: PygameSurface,
    center: tuple[int, int],
    radius: int,
    start: float,
    end: float,
    color: Color,
    width: int,
) -> None:
    style = SPEED_GAUGE_STYLE
    if end <= start:
        return
    segment_count = max(1, round(style.arc_segments * (end - start)))
    previous = _arc_point(center=center, radius=radius, normalized=start)
    for segment in range(1, segment_count + 1):
        normalized = start + ((end - start) * segment / segment_count)
        current = _arc_point(center=center, radius=radius, normalized=normalized)
        pygame.draw.line(surface, color, previous, current, width=width)
        previous = current


def _draw_ticks(
    *,
    pygame: PygameModule,
    surface: PygameSurface,
    center: tuple[int, int],
    radius: int,
    scale: int,
) -> None:
    style = SPEED_GAUGE_STYLE
    for index in range(style.tick_count):
        normalized = index / max(1, style.tick_count - 1)
        outer = _arc_point(center=center, radius=radius + (2 * scale), normalized=normalized)
        inner = _arc_point(center=center, radius=radius - (5 * scale), normalized=normalized)
        pygame.draw.line(surface, style.tick, outer, inner, width=max(1, scale))


def _draw_needle(
    *,
    pygame: PygameModule,
    surface: PygameSurface,
    center: tuple[int, int],
    radius: int,
    normalized: float,
    scale: int,
) -> None:
    style = SPEED_GAUGE_STYLE
    tip = _arc_point(
        center=center,
        radius=round(radius * style.needle_length_fraction),
        normalized=normalized,
    )
    shadow_tip = (tip[0] + scale, tip[1] + scale)
    shadow_center = (center[0] + scale, center[1] + scale)
    pygame.draw.line(surface, style.needle_shadow, shadow_center, shadow_tip, width=3 * scale)
    pygame.draw.line(surface, style.needle, center, tip, width=style.needle_width * scale)
    pygame.draw.circle(surface, style.pivot, center, 4 * scale)
    pygame.draw.circle(surface, style.needle, center, 2 * scale)


def _draw_speed_text(
    *,
    screen: PygameSurface,
    rect: PygameRect,
    speed_kph: float | None,
    value_font: RenderFont,
) -> None:
    style = SPEED_GAUGE_STYLE
    label = (
        "---" if speed_kph is None or not math.isfinite(speed_kph) else f"{round(speed_kph):04d}"
    )
    value_surface = value_font.render(f"{label} kph", True, PALETTE.text_primary)
    screen.blit(
        value_surface,
        (
            rect.centerx - (value_surface.get_width() // 2),
            rect.bottom - style.label_y_offset,
        ),
    )


def _draw_energy_meter(
    *,
    pygame: PygameModule,
    screen: PygameSurface,
    rect: PygameRect,
    energy_fraction: float | None,
) -> None:
    style = SPEED_GAUGE_STYLE
    meter_height = min(style.energy_meter_height, max(0, rect.height - 12))
    meter = pygame.Rect(
        rect.right - style.energy_meter_margin_right - style.energy_meter_width,
        rect.centery - (meter_height // 2) + style.energy_meter_center_y_offset,
        style.energy_meter_width,
        meter_height,
    )
    radius = style.energy_meter_radius
    pygame.draw.rect(screen, style.energy_track, meter, border_radius=radius)
    pygame.draw.rect(
        screen,
        style.energy_border,
        meter,
        width=style.energy_outer_border_width,
        border_radius=radius,
    )
    inner = meter.inflate(-(2 * style.energy_meter_padding), -(2 * style.energy_meter_padding))
    if inner.width <= 0 or inner.height <= 0:
        return

    pygame.draw.rect(screen, style.energy_empty_fill, inner, border_radius=max(1, radius - 2))
    if energy_fraction is not None:
        fill_height = _energy_fill_height(
            energy_fraction=energy_fraction, meter_height=inner.height
        )
        if fill_height > 0:
            fill = pygame.Rect(inner.left, inner.bottom - fill_height, inner.width, fill_height)
            pygame.draw.rect(screen, _energy_color(energy_fraction), fill, border_radius=2)
            pygame.draw.line(
                screen,
                style.energy_fill_lip,
                fill.topleft,
                fill.topright,
                width=1,
            )
    _draw_energy_glass(pygame=pygame, screen=screen, meter=meter)
    pygame.draw.rect(
        screen,
        style.energy_inner_border,
        inner,
        width=style.energy_inner_border_width,
        border_radius=max(1, radius - 2),
    )
    pygame.draw.rect(
        screen,
        style.energy_border,
        meter,
        width=style.energy_outer_border_width,
        border_radius=radius,
    )


def _energy_fill_height(*, energy_fraction: float, meter_height: int) -> int:
    clamped = max(0.0, min(1.0, energy_fraction))
    if clamped <= 0.0 or meter_height <= 0:
        return 0
    return max(1, min(meter_height, round(clamped * meter_height)))


def _draw_energy_glass(
    *,
    pygame: PygameModule,
    screen: PygameSurface,
    meter: PygameRect,
) -> None:
    style = SPEED_GAUGE_STYLE
    overlay = pygame.Surface(meter.size, pygame.SRCALPHA)
    overlay_rect = overlay.get_rect()
    shadow = pygame.Rect(
        0,
        overlay_rect.centery,
        overlay_rect.width,
        overlay_rect.height - overlay_rect.centery,
    )
    highlight = pygame.Rect(2, 2, max(1, overlay_rect.width // 3), max(1, overlay_rect.height - 4))
    pygame.draw.rect(
        overlay, style.energy_glass_shadow, shadow, border_radius=style.energy_meter_radius
    )
    pygame.draw.rect(overlay, style.energy_glass_highlight, highlight, border_radius=2)
    pygame.draw.line(
        overlay,
        style.energy_glass_edge,
        (2, 1),
        (max(2, overlay_rect.width - 3), 1),
        width=1,
    )
    screen.blit(overlay, meter.topleft)


def _energy_color(energy_fraction: float) -> Color:
    return (
        SPEED_GAUGE_STYLE.energy_low_fill
        if energy_fraction < 0.25
        else SPEED_GAUGE_STYLE.energy_fill
    )


def _arc_point(
    *,
    center: tuple[int, int],
    radius: int,
    normalized: float,
) -> tuple[int, int]:
    style = SPEED_GAUGE_STYLE
    degrees = style.start_degrees - (normalized * style.sweep_degrees)
    radians = math.radians(degrees)
    return (
        center[0] + round(math.cos(radians) * radius),
        center[1] - round(math.sin(radians) * radius),
    )


def _speed_color(normalized: float) -> Color:
    return (
        SPEED_GAUGE_STYLE.active_high
        if normalized >= SPEED_GAUGE_STYLE.red_zone_start
        else SPEED_GAUGE_STYLE.active_low
    )
