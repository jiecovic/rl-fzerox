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
STEER_EXTREME_COLOR: Color = PALETTE.text_accent
COCKPIT_PANEL_FILL: Color = (14, 19, 25)
COCKPIT_PANEL_BORDER: Color = (74, 91, 112)
COCKPIT_GRID: Color = (31, 42, 54)
THRUST_WARNING_FILL: Color = (241, 206, 108)
THRUST_COLUMN_BORDER_WIDTH = 2
LEAN_ACTIVE_FILL: Color = (27, 82, 66)
LEAN_ACTIVE_BORDER: Color = (126, 214, 170)
BUTTON_SHADOW: Color = (7, 10, 14)
BUTTON_FACE_FILL: Color = (24, 31, 39)
BUTTON_FACE_HIGHLIGHT: Color = (62, 76, 92)
BUTTON_FACE_INNER: Color = (17, 23, 30)
BUTTON_ACTIVE_TEXT: Color = (232, 255, 246)
GLASS_HIGHLIGHT: Color = (170, 190, 210)
GLASS_LOWLIGHT: Color = (8, 11, 15)
GLASS_SHEEN: tuple[int, int, int, int] = (238, 248, 255, 82)
GLASS_SHADOW: tuple[int, int, int, int] = (0, 0, 0, 44)
GLASS_EDGE_GLOW: tuple[int, int, int, int] = (126, 214, 170, 48)
BOOST_EDGE_GLOW: tuple[int, int, int, int] = (126, 214, 170, 52)


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

    lean_y = panel.y + 76
    lean_width = 48 if dual_gas_levers else 56
    lean_gap = 6 if dual_gas_levers else 8
    boost_radius = 11 if dual_gas_levers else 13
    boost_center_x = steer_x + lean_width + lean_gap + boost_radius
    right_lean_x = boost_center_x + boost_radius + lean_gap
    boost_center_y = lean_y + ((_pill_height(fonts.small) + 2) // 2)
    _draw_lean_button(
        pygame=pygame,
        screen=screen,
        font=fonts.small,
        x=steer_x,
        y=lean_y,
        width=lean_width,
        direction=-1,
        active=control_viz.lean_direction < 0,
    )
    _draw_boost_button(
        pygame=pygame,
        screen=screen,
        center=(boost_center_x, boost_center_y),
        radius=boost_radius,
        level=control_viz.boost_lamp_level,
    )
    _draw_lean_button(
        pygame=pygame,
        screen=screen,
        font=fonts.small,
        x=right_lean_x,
        y=lean_y,
        width=lean_width,
        direction=1,
        active=control_viz.lean_direction > 0,
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

    segment_count = 7
    segment_gap = 1
    segment_height = 8
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


def _draw_lean_button(
    *,
    pygame,
    screen,
    font,
    x: int,
    y: int,
    width: int,
    direction: int,
    active: bool,
) -> None:
    height = _pill_height(font) + 2
    rect = pygame.Rect(x, y, width, height)
    points = _lean_button_points(rect, direction=direction)
    inner_points = _lean_button_points(rect.inflate(-5, -5), direction=direction)
    pygame.draw.polygon(screen, BUTTON_SHADOW, _offset_points(points, dx=1, dy=2))

    fill_color = LEAN_ACTIVE_FILL if active else BUTTON_FACE_FILL
    border_color = LEAN_ACTIVE_BORDER if active else PALETTE.flag_inactive_border
    text_color = BUTTON_ACTIVE_TEXT if active else PALETTE.text_muted
    pygame.draw.polygon(screen, fill_color, points)
    pygame.draw.polygon(screen, BUTTON_FACE_INNER if not active else LEAN_ACTIVE_FILL, inner_points)
    _draw_alpha_polygon(
        pygame=pygame,
        screen=screen,
        points=_lean_button_sheen_points(rect, direction=direction),
        color=GLASS_SHEEN,
    )
    _draw_alpha_polygon(
        pygame=pygame,
        screen=screen,
        points=_lean_button_shadow_points(rect, direction=direction),
        color=GLASS_SHADOW,
    )
    if active:
        _draw_alpha_polygon(
            pygame=pygame,
            screen=screen,
            points=_offset_points(points, dx=0, dy=0),
            color=GLASS_EDGE_GLOW,
        )
        pygame.draw.polygon(screen, fill_color, inner_points)
        _draw_alpha_polygon(
            pygame=pygame,
            screen=screen,
            points=_lean_button_sheen_points(rect, direction=direction),
            color=GLASS_SHEEN,
        )
        _draw_alpha_polygon(
            pygame=pygame,
            screen=screen,
            points=_lean_button_shadow_points(rect, direction=direction),
            color=GLASS_SHADOW,
        )
    pygame.draw.polygon(screen, border_color, points, width=2 if active else 1)

    highlight_start = (rect.left + 10, rect.top + 4)
    highlight_end = (rect.right - 10, rect.top + 4)
    pygame.draw.line(
        screen,
        LEAN_ACTIVE_BORDER if active else BUTTON_FACE_HIGHLIGHT,
        highlight_start,
        highlight_end,
        width=1,
    )

    arrow_color = LEAN_ACTIVE_BORDER if active else PALETTE.text_muted
    arrow_center_y = rect.centery
    if direction < 0:
        arrow_points = (
            (rect.left + 8, arrow_center_y),
            (rect.left + 16, arrow_center_y - 5),
            (rect.left + 16, arrow_center_y + 5),
        )
        label_x_offset = 7
    else:
        arrow_points = (
            (rect.right - 8, arrow_center_y),
            (rect.right - 16, arrow_center_y - 5),
            (rect.right - 16, arrow_center_y + 5),
        )
        label_x_offset = -7
    pygame.draw.polygon(screen, arrow_color, arrow_points)

    label = "L" if direction < 0 else "R"
    label_surface = font.render(label, True, text_color)
    screen.blit(
        label_surface,
        (
            rect.centerx - (label_surface.get_width() // 2) + label_x_offset,
            rect.centery - (label_surface.get_height() // 2),
        ),
    )


def _lean_button_points(rect, *, direction: int) -> tuple[tuple[int, int], ...]:
    inset = 12
    if direction < 0:
        return (
            (rect.left, rect.centery),
            (rect.left + inset, rect.top),
            (rect.right, rect.top),
            (rect.right - 5, rect.centery),
            (rect.right, rect.bottom),
            (rect.left + inset, rect.bottom),
        )
    return (
        (rect.right, rect.centery),
        (rect.right - inset, rect.top),
        (rect.left, rect.top),
        (rect.left + 5, rect.centery),
        (rect.left, rect.bottom),
        (rect.right - inset, rect.bottom),
    )


def _lean_button_sheen_points(rect, *, direction: int) -> tuple[tuple[int, int], ...]:
    if direction < 0:
        return (
            (rect.left + 14, rect.top + 3),
            (rect.right - 5, rect.top + 3),
            (rect.right - 9, rect.centery - 3),
            (rect.left + 18, rect.centery - 3),
            (rect.left + 8, rect.centery),
        )
    return (
        (rect.right - 14, rect.top + 3),
        (rect.left + 5, rect.top + 3),
        (rect.left + 9, rect.centery - 3),
        (rect.right - 18, rect.centery - 3),
        (rect.right - 8, rect.centery),
    )


def _lean_button_shadow_points(rect, *, direction: int) -> tuple[tuple[int, int], ...]:
    if direction < 0:
        return (
            (rect.left + 8, rect.centery + 2),
            (rect.right - 8, rect.centery + 2),
            (rect.right - 3, rect.bottom - 3),
            (rect.left + 14, rect.bottom - 3),
        )
    return (
        (rect.right - 8, rect.centery + 2),
        (rect.left + 8, rect.centery + 2),
        (rect.left + 3, rect.bottom - 3),
        (rect.right - 14, rect.bottom - 3),
    )


def _draw_boost_button(
    *,
    pygame,
    screen,
    center: tuple[int, int],
    radius: int,
    level: float,
) -> None:
    level = max(0.0, min(1.0, level))
    active = level > 0.0
    manual_intensity = max(0.0, min(1.0, (level - 0.55) / 0.45))
    normal_intensity = max(0.0, min(1.0, level / 0.55))
    manual_dominance = manual_intensity**0.45
    pygame.draw.circle(screen, BUTTON_SHADOW, (center[0] + 1, center[1] + 2), radius)
    if active:
        _draw_alpha_circle(
            pygame=pygame,
            screen=screen,
            center=center,
            radius=radius + 5 + round(4 * manual_intensity),
            color=(
                round(150 + (40 * normal_intensity) - (72 * manual_dominance)),
                255,
                round(190 + (28 * normal_intensity) - (126 * manual_dominance)),
                round((34 * normal_intensity) + (88 * manual_intensity)),
            ),
        )
    if manual_intensity > 0.0:
        _draw_alpha_circle(
            pygame=pygame,
            screen=screen,
            center=center,
            radius=radius + 9,
            color=(78, 255, 64, round(78 * manual_intensity)),
        )
    bezel_color = _blend_color(BUTTON_FACE_FILL, (29, 58, 46), normal_intensity)
    border_color = _blend_color(
        PALETTE.flag_inactive_border,
        (132, 214, 172),
        normal_intensity,
    )
    led_outer = _blend_color((37, 60, 52), (92, 176, 130), normal_intensity)
    led_outer = _blend_color(led_outer, (34, 255, 44), manual_dominance)
    led_inner = _blend_color((50, 86, 70), (174, 234, 196), normal_intensity)
    led_inner = _blend_color(led_inner, (82, 255, 26), manual_dominance)
    pygame.draw.circle(screen, bezel_color, center, radius)
    pygame.draw.circle(screen, led_outer, center, max(1, radius - 4))
    pygame.draw.circle(screen, led_inner, center, max(1, radius - 7))
    if manual_intensity > 0.0:
        flash_core = _blend_color((90, 255, 52), (34, 255, 0), manual_dominance)
        pygame.draw.circle(screen, flash_core, center, max(1, radius - 9))
    _draw_alpha_circle(
        pygame=pygame,
        screen=screen,
        center=(center[0] - max(2, radius // 4), center[1] - max(2, radius // 4)),
        radius=max(3, radius // 2),
        color=GLASS_SHEEN,
    )
    _draw_alpha_circle(
        pygame=pygame,
        screen=screen,
        center=(center[0] + 1, center[1] + max(3, radius // 3)),
        radius=max(3, radius // 2),
        color=GLASS_SHADOW,
    )
    if active:
        _draw_alpha_circle(
            pygame=pygame,
            screen=screen,
            center=center,
            radius=radius + 2,
            color=(
                BOOST_EDGE_GLOW[0],
                BOOST_EDGE_GLOW[1],
                BOOST_EDGE_GLOW[2],
                round(24 + (58 * level) + (68 * manual_intensity)),
            ),
        )
        pygame.draw.circle(screen, led_outer, center, max(1, radius - 4))
        pygame.draw.circle(screen, led_inner, center, max(1, radius - 7))
        _draw_alpha_circle(
            pygame=pygame,
            screen=screen,
            center=(center[0] - max(2, radius // 4), center[1] - max(2, radius // 4)),
            radius=max(3, radius // 2),
            color=GLASS_SHEEN,
        )
    pygame.draw.circle(screen, border_color, center, radius, width=2 if level >= 0.75 else 1)
    _draw_alpha_circle(
        pygame=pygame,
        screen=screen,
        center=(center[0] - 3, center[1] - 4),
        radius=max(2, radius // 4),
        color=(255, 255, 255, 90),
    )


def _blend_color(start: Color, end: Color, weight: float) -> Color:
    weight = max(0.0, min(1.0, weight))
    return (
        round(start[0] + ((end[0] - start[0]) * weight)),
        round(start[1] + ((end[1] - start[1]) * weight)),
        round(start[2] + ((end[2] - start[2]) * weight)),
    )


def _offset_points(
    points: tuple[tuple[int, int], ...],
    *,
    dx: int,
    dy: int,
) -> tuple[tuple[int, int], ...]:
    return tuple((x + dx, y + dy) for x, y in points)


def _draw_alpha_polygon(
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


def _draw_alpha_circle(
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


def _draw_glass_track_overlay(*, pygame, screen, track) -> None:
    _draw_alpha_rect(
        pygame=pygame,
        screen=screen,
        rect=pygame.Rect(track.left + 4, track.top + 2, max(1, track.width - 8), 3),
        color=(255, 255, 255, 46),
    )
    _draw_alpha_rect(
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
    _draw_alpha_rect(
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


def _draw_glass_column_overlay(*, pygame, screen, track) -> None:
    _draw_alpha_rect(
        pygame=pygame,
        screen=screen,
        rect=pygame.Rect(track.left + 3, track.top + 4, 4, max(1, track.height - 8)),
        color=(255, 255, 255, 78),
    )
    _draw_alpha_rect(
        pygame=pygame,
        screen=screen,
        rect=pygame.Rect(track.left + 4, track.top + 3, max(1, track.width - 8), track.height // 2),
        color=(255, 255, 255, 38),
    )
    _draw_alpha_rect(
        pygame=pygame,
        screen=screen,
        rect=pygame.Rect(track.centerx, track.top + 5, 2, max(1, track.height - 10)),
        color=(255, 255, 255, 34),
    )
    _draw_alpha_rect(
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


def _draw_alpha_rect(*, pygame, screen, rect, color: tuple[int, int, int, int]) -> None:
    surface = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
    surface.fill(color)
    screen.blit(surface, rect.topleft)


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
