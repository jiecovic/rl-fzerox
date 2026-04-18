# src/rl_fzerox/ui/watch/render/widgets.py
from __future__ import annotations

from dataclasses import dataclass

from rl_fzerox.ui.watch.layout import (
    LAYOUT,
    PALETTE,
    Color,
    ControlViz,
    FlagViz,
    ViewerFonts,
)


@dataclass(frozen=True)
class SteerAxisGuide:
    """Approximate F-Zero X steering thresholds in policy action space."""

    deadzone: float = 0.08
    saturation: float = 63.0 / 80.0


STEER_AXIS_GUIDE = SteerAxisGuide()
STEER_DEADZONE_COLOR: Color = (96, 165, 250)
COCKPIT_PANEL_FILL: Color = (14, 19, 25)
COCKPIT_PANEL_BORDER: Color = (74, 91, 112)
COCKPIT_GRID: Color = (31, 42, 54)
THRUST_WARNING_FILL: Color = (241, 206, 108)


def _draw_round_marker(*, pygame, screen, color, center, radius: int, outline_color) -> None:
    gfxdraw = getattr(pygame, "gfxdraw", None)
    if gfxdraw is not None:
        gfxdraw.filled_circle(screen, center[0], center[1], radius, color)
        gfxdraw.aacircle(screen, center[0], center[1], radius, color)
        if outline_color is not None and radius > 1:
            gfxdraw.aacircle(screen, center[0], center[1], radius, outline_color)
        return

    pygame.draw.circle(screen, color, center, radius)
    if outline_color is not None and radius > 1:
        pygame.draw.circle(screen, outline_color, center, radius, width=1)


def _draw_flag_viz(
    *,
    pygame,
    screen,
    fonts: ViewerFonts,
    x: int,
    y: int,
    flag_viz: FlagViz,
) -> int:
    label_surface = fonts.small.render("Flags", True, PALETTE.text_muted)
    screen.blit(label_surface, (x, y))
    y += label_surface.get_height() + LAYOUT.line_gap

    pill_height = 0
    for row in flag_viz.rows:
        row_x = x
        for token in row:
            pill_rect = _draw_pill(
                pygame=pygame,
                screen=screen,
                font=fonts.small,
                x=row_x,
                y=y,
                label=token.label,
                active=token.active,
                active_text_color=PALETTE.text_primary,
                active_fill_color=PALETTE.flag_active_background,
                active_border_color=PALETTE.flag_active_border,
            )
            pill_height = pill_rect.height
            row_x += pill_rect.width + LAYOUT.flag_token_gap
        y += pill_height + LAYOUT.line_gap

    return y


def _draw_control_viz(
    *,
    pygame,
    screen,
    fonts: ViewerFonts,
    x: int,
    y: int,
    width: int,
    control_viz: ControlViz,
) -> int:
    panel_height = 118
    panel = pygame.Rect(x, y, width, panel_height)
    _draw_cockpit_panel(pygame=pygame, screen=screen, rect=panel)

    header = fonts.small.render("COCKPIT CONTROL", True, PALETTE.text_muted)
    screen.blit(header, (panel.x + 12, panel.y + 7))

    dual_gas_levers = control_viz.air_brake_axis is not None
    thrust_gap = 22 if dual_gas_levers else 0
    thrust_group_width = (
        (2 * LAYOUT.control_gas_width) + thrust_gap if dual_gas_levers else LAYOUT.control_gas_width
    )
    thrust_x = panel.right - 28 - thrust_group_width
    air_brake_x = thrust_x + LAYOUT.control_gas_width + thrust_gap
    thrust_y = panel.y + 28

    _draw_centered_label(
        screen=screen,
        font=fonts.small,
        label="THR" if dual_gas_levers else "THRUST",
        color=PALETTE.text_muted,
        center_x=thrust_x + (LAYOUT.control_gas_width // 2),
        y=panel.y + 8,
    )
    if dual_gas_levers:
        _draw_centered_label(
            screen=screen,
            font=fonts.small,
            label="BRK",
            color=PALETTE.text_muted,
            center_x=air_brake_x + (LAYOUT.control_gas_width // 2),
            y=panel.y + 8,
        )

    steer_x = panel.x + 14
    steer_width = max(84, thrust_x - steer_x - 22)
    steer_track = pygame.Rect(
        steer_x,
        panel.y + 42,
        steer_width,
        LAYOUT.control_steer_height,
    )
    _draw_steer_instrument(
        pygame=pygame,
        screen=screen,
        track=steer_track,
        value=control_viz.steer_x,
    )

    steer_value = max(-1.0, min(1.0, control_viz.steer_x))
    steer_text_color = (
        PALETTE.text_warning
        if abs(steer_value) >= STEER_AXIS_GUIDE.saturation
        else STEER_DEADZONE_COLOR
        if abs(steer_value) <= STEER_AXIS_GUIDE.deadzone
        else PALETTE.text_primary
    )
    steer_readout = fonts.body.render(f"steer {steer_value:+.2f}", True, steer_text_color)
    screen.blit(steer_readout, (steer_x, steer_track.bottom + 7))

    lean_y = panel.y + 76
    _draw_lean_chevron(
        pygame=pygame,
        screen=screen,
        font=fonts.small,
        x=steer_x,
        y=lean_y,
        direction=-1,
        active=control_viz.lean_direction < 0,
    )
    _draw_lean_chevron(
        pygame=pygame,
        screen=screen,
        font=fonts.small,
        x=steer_x + 72,
        y=lean_y,
        direction=1,
        active=control_viz.lean_direction > 0,
    )
    _draw_hex_button(
        pygame=pygame,
        screen=screen,
        font=fonts.small,
        rect=pygame.Rect(steer_x + 146, lean_y, 64, _pill_height(fonts.small) + 2),
        label="boost",
        active=control_viz.boost_pressed,
    )

    gas_level = max(0.0, min(1.0, control_viz.gas_level))
    thrust_color = (
        THRUST_WARNING_FILL
        if (
            control_viz.thrust_warning_threshold is not None
            and gas_level < control_viz.thrust_warning_threshold
        )
        else PALETTE.text_accent
    )
    _draw_thrust_column(
        pygame=pygame,
        screen=screen,
        x=thrust_x,
        y=thrust_y,
        level=gas_level,
        threshold=control_viz.thrust_warning_threshold,
        fill_color=thrust_color,
    )
    if dual_gas_levers:
        air_brake_level = max(0.0, min(1.0, control_viz.air_brake_axis or 0.0))
        air_brake_color = (
            PALETTE.text_muted if control_viz.air_brake_disabled else PALETTE.text_warning
        )
        _draw_thrust_column(
            pygame=pygame,
            screen=screen,
            x=air_brake_x,
            y=thrust_y,
            level=air_brake_level,
            threshold=None,
            fill_color=air_brake_color,
        )

    percent = fonts.body.render(
        f"{round(gas_level * 100):3d}%",
        True,
        thrust_color if gas_level > 0.0 else PALETTE.text_muted,
    )
    percent_x = thrust_x + (thrust_group_width // 2) - (percent.get_width() // 2)
    screen.blit(percent, (percent_x, thrust_y + LAYOUT.control_gas_height + 4))
    return panel.bottom


def _draw_cockpit_panel(*, pygame, screen, rect) -> None:
    cut = 10
    points = (
        (rect.left + cut, rect.top),
        (rect.right - cut, rect.top),
        (rect.right, rect.top + cut),
        (rect.right, rect.bottom - cut),
        (rect.right - cut, rect.bottom),
        (rect.left + cut, rect.bottom),
        (rect.left, rect.bottom - cut),
        (rect.left, rect.top + cut),
    )
    pygame.draw.polygon(screen, COCKPIT_PANEL_FILL, points)
    for x_offset in range(18, rect.width, 32):
        pygame.draw.line(
            screen,
            COCKPIT_GRID,
            (rect.left + x_offset, rect.top + 1),
            (rect.left + x_offset - 28, rect.bottom - 1),
        )
    pygame.draw.polygon(screen, COCKPIT_PANEL_BORDER, points, width=1)


def _draw_steer_instrument(*, pygame, screen, track, value: float) -> None:
    pygame.draw.rect(
        screen,
        PALETTE.control_track,
        track,
        border_radius=track.height // 2,
    )
    _draw_steer_axis_guides(pygame=pygame, screen=screen, track=track)
    _draw_steer_fill(pygame=pygame, screen=screen, track=track, value=value)
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
        color=PALETTE.control_knob,
        center=(_steer_track_x(track=track, value=value), track.centery),
        radius=LAYOUT.control_marker_radius,
        outline_color=PALETTE.control_knob_outline,
    )


def _draw_thrust_column(
    *,
    pygame,
    screen,
    x: int,
    y: int,
    level: float,
    threshold: float | None,
    fill_color,
) -> None:
    level = max(0.0, min(1.0, level))
    track = pygame.Rect(
        x,
        y,
        LAYOUT.control_gas_width,
        LAYOUT.control_gas_height,
    )
    pygame.draw.rect(
        screen,
        PALETTE.control_track,
        track,
        border_radius=3,
    )

    segment_count = 7
    segment_gap = 2
    segment_height = max(
        2,
        (track.height - ((segment_count + 1) * segment_gap)) // segment_count,
    )
    lit_segments = round(level * segment_count)
    for segment_index in range(segment_count):
        segment_y = track.bottom - segment_gap - ((segment_index + 1) * segment_height)
        segment_y -= segment_index * segment_gap
        segment = pygame.Rect(track.x + 3, segment_y, track.width - 6, segment_height)
        if segment_index < lit_segments:
            pygame.draw.rect(screen, fill_color, segment, border_radius=1)
        else:
            pygame.draw.rect(screen, COCKPIT_GRID, segment, border_radius=1)

    if threshold is not None:
        threshold = max(0.0, min(1.0, threshold))
        threshold_y = track.bottom - round(track.height * threshold)
        pygame.draw.line(
            screen,
            THRUST_WARNING_FILL,
            (track.left - 4, threshold_y),
            (track.right + 4, threshold_y),
            width=2,
        )

    pygame.draw.rect(
        screen,
        PALETTE.control_lever_border,
        track,
        width=PALETTE.control_lever_border_width,
        border_radius=3,
    )


def _draw_lean_chevron(
    *,
    pygame,
    screen,
    font,
    x: int,
    y: int,
    direction: int,
    active: bool,
) -> None:
    width = 56
    height = _pill_height(font) + 2
    rect = pygame.Rect(x, y, width, height)
    inset = 11
    if direction < 0:
        points = (
            (rect.left, rect.centery),
            (rect.right - inset, rect.top),
            (rect.right, rect.top),
            (rect.right, rect.bottom),
            (rect.right - inset, rect.bottom),
        )
    else:
        points = (
            (rect.right, rect.centery),
            (rect.left + inset, rect.top),
            (rect.left, rect.top),
            (rect.left, rect.bottom),
            (rect.left + inset, rect.bottom),
        )
    fill_color = PALETTE.flag_active_background if active else PALETTE.flag_inactive_background
    border_color = PALETTE.flag_active_border if active else PALETTE.flag_inactive_border
    text_color = PALETTE.text_primary if active else PALETTE.text_muted
    pygame.draw.polygon(screen, fill_color, points)
    pygame.draw.polygon(screen, border_color, points, width=1)
    label = "lean"
    text_surface = font.render(label, True, text_color)
    screen.blit(
        text_surface,
        (
            rect.centerx - (text_surface.get_width() // 2),
            rect.centery - (text_surface.get_height() // 2),
        ),
    )


def _draw_hex_button(
    *,
    pygame,
    screen,
    font,
    rect,
    label: str,
    active: bool,
) -> None:
    cut = min(7, rect.height // 3)
    points = (
        (rect.left + cut, rect.top),
        (rect.right - cut, rect.top),
        (rect.right, rect.centery),
        (rect.right - cut, rect.bottom),
        (rect.left + cut, rect.bottom),
        (rect.left, rect.centery),
    )
    fill_color = PALETTE.text_warning if active else PALETTE.flag_inactive_background
    border_color = PALETTE.text_warning if active else PALETTE.flag_inactive_border
    text_color = PALETTE.panel_background if active else PALETTE.text_muted
    pygame.draw.polygon(screen, fill_color, points)
    pygame.draw.polygon(screen, border_color, points, width=1)
    text_surface = font.render(label, True, text_color)
    screen.blit(
        text_surface,
        (
            rect.centerx - (text_surface.get_width() // 2),
            rect.centery - (text_surface.get_height() // 2),
        ),
    )


def _draw_steer_fill(*, pygame, screen, track, value: float) -> None:
    value = max(-1.0, min(1.0, value))
    magnitude = abs(value)
    if magnitude == 0.0:
        return

    fill_color = (
        STEER_DEADZONE_COLOR if magnitude <= STEER_AXIS_GUIDE.deadzone else PALETTE.text_primary
    )
    normal_magnitude = min(magnitude, STEER_AXIS_GUIDE.saturation)
    _draw_steer_segment(
        pygame=pygame,
        screen=screen,
        track=track,
        start=0.0,
        end=normal_magnitude if value > 0.0 else -normal_magnitude,
        color=fill_color,
    )
    if magnitude <= STEER_AXIS_GUIDE.saturation:
        return

    _draw_steer_segment(
        pygame=pygame,
        screen=screen,
        track=track,
        start=STEER_AXIS_GUIDE.saturation if value > 0.0 else -STEER_AXIS_GUIDE.saturation,
        end=value,
        color=PALETTE.text_warning,
    )


def _draw_steer_segment(*, pygame, screen, track, start: float, end: float, color) -> None:
    start_x = _steer_track_x(track=track, value=start)
    end_x = _steer_track_x(track=track, value=end)
    segment_x = min(start_x, end_x)
    segment_width = max(1, abs(end_x - start_x))
    pygame.draw.rect(
        screen,
        color,
        pygame.Rect(segment_x, track.y, segment_width, track.height),
        border_radius=track.height // 2,
    )


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
            PALETTE.text_warning,
            (marker_x, marker_top),
            (marker_x, marker_bottom),
        )


def _steer_track_x(*, track, value: float) -> int:
    value = max(-1.0, min(1.0, value))
    return track.centerx + round((track.width // 2) * value)


def _draw_centered_label(*, screen, font, label: str, color, center_x: int, y: int) -> None:
    surface = font.render(label, True, color)
    screen.blit(surface, (center_x - (surface.get_width() // 2), y))


def _pill_width(font, label: str) -> int:
    return font.render(label, True, PALETTE.text_primary).get_width() + (
        2 * LAYOUT.flag_token_pad_x
    )


def _pill_height(font) -> int:
    return font.render("Ag", True, PALETTE.text_primary).get_height() + (
        2 * LAYOUT.flag_token_pad_y
    )


def _draw_pill(
    *,
    pygame,
    screen,
    font,
    x: int,
    y: int,
    label: str,
    active: bool,
    active_text_color,
    active_fill_color,
    active_border_color,
):
    text_color = active_text_color if active else PALETTE.text_muted
    fill_color = active_fill_color if active else PALETTE.flag_inactive_background
    border_color = active_border_color if active else PALETTE.flag_inactive_border
    text_surface = font.render(label, True, text_color)
    pill_rect = pygame.Rect(
        x,
        y,
        text_surface.get_width() + (2 * LAYOUT.flag_token_pad_x),
        text_surface.get_height() + (2 * LAYOUT.flag_token_pad_y),
    )
    pygame.draw.rect(
        screen,
        fill_color,
        pill_rect,
        border_radius=pill_rect.height // 2,
    )
    pygame.draw.rect(
        screen,
        border_color,
        pill_rect,
        width=1,
        border_radius=pill_rect.height // 2,
    )
    screen.blit(
        text_surface,
        (x + LAYOUT.flag_token_pad_x, y + LAYOUT.flag_token_pad_y),
    )
    return pill_rect
