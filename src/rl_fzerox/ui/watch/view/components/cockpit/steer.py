# src/rl_fzerox/ui/watch/view/components/cockpit/steer.py
from __future__ import annotations

from rl_fzerox.ui.watch.view.components.cockpit.primitives import _draw_round_marker
from rl_fzerox.ui.watch.view.components.cockpit.style import (
    STEER_AXIS_GUIDE,
    STEER_DEADZONE_COLOR,
    STEER_EXTREME_COLOR,
)
from rl_fzerox.ui.watch.view.components.effects import (
    draw_glass_track_overlay as _draw_glass_track_overlay,
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
    pygame.draw.rect(
        screen,
        PALETTE.control_track,
        track,
        border_radius=track.height // 2,
    )
    _draw_steer_axis_guides(pygame=pygame, screen=screen, track=track)
    _draw_steer_fill(pygame=pygame, screen=screen, track=track, value=value)
    _draw_glass_track_overlay(pygame=pygame, screen=screen, track=track)
    pygame.draw.rect(
        screen,
        PALETTE.control_lever_border,
        track,
        width=PALETTE.control_lever_border_width,
        border_radius=track.height // 2,
    )
    _draw_round_marker(
        pygame=pygame,
        screen=screen,
        color=_steer_indicator_color(value),
        center=(_steer_track_x(track=track, value=value), track.centery),
        radius=marker_radius,
        outline_color=PALETTE.control_knob_outline,
    )


def _draw_steer_fill(*, pygame, screen, track, value: float) -> None:
    value = max(-1.0, min(1.0, value))
    magnitude = abs(value)
    if magnitude == 0.0:
        return

    fill_color = _steer_indicator_color(value)
    _draw_steer_segment(
        pygame=pygame,
        screen=screen,
        track=track,
        start=0.0,
        end=value,
        color=fill_color,
    )


def _draw_steer_segment(*, pygame, screen, track, start: float, end: float, color) -> None:
    start_x = _steer_track_x(track=track, value=start)
    end_x = _steer_track_x(track=track, value=end)
    segment_x = min(start_x, end_x)
    segment_width = max(1, abs(end_x - start_x))
    segment_rect = pygame.Rect(segment_x, track.y, segment_width, track.height)
    pygame.draw.rect(
        screen,
        color,
        segment_rect,
        border_radius=track.height // 2,
    )


def _steer_indicator_color(value: float) -> Color:
    magnitude = abs(max(-1.0, min(1.0, value)))
    if magnitude <= STEER_AXIS_GUIDE.deadzone:
        return STEER_DEADZONE_COLOR
    if magnitude >= STEER_AXIS_GUIDE.saturation:
        return STEER_EXTREME_COLOR
    return PALETTE.text_primary


def _draw_steer_axis_guides(*, pygame, screen, track) -> None:
    marker_top = track.top - 2
    marker_bottom = track.bottom + 2
    center_x = _steer_track_x(track=track, value=0.0)
    pygame.draw.line(
        screen,
        PALETTE.flag_inactive_border,
        (center_x, marker_top),
        (center_x, marker_bottom),
    )
    for value in (-STEER_AXIS_GUIDE.deadzone, STEER_AXIS_GUIDE.deadzone):
        marker_x = _steer_track_x(track=track, value=value)
        pygame.draw.line(
            screen,
            STEER_DEADZONE_COLOR,
            (marker_x, marker_top),
            (marker_x, marker_bottom),
        )
    for value in (-STEER_AXIS_GUIDE.saturation, STEER_AXIS_GUIDE.saturation):
        marker_x = _steer_track_x(track=track, value=value)
        pygame.draw.line(
            screen,
            STEER_EXTREME_COLOR,
            (marker_x, marker_top),
            (marker_x, marker_bottom),
        )


def _steer_track_x(*, track, value: float) -> int:
    value = max(-1.0, min(1.0, value))
    return track.centerx + round((track.width // 2) * value)
