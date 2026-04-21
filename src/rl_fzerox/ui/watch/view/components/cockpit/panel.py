# src/rl_fzerox/ui/watch/view/components/cockpit/panel.py
from __future__ import annotations

from rl_fzerox.ui.watch.view.components.cockpit.axis import (
    draw_pitch_instrument,
    draw_steer_instrument,
)
from rl_fzerox.ui.watch.view.components.cockpit.buttons import (
    _draw_boost_button,
    _draw_lean_button,
)
from rl_fzerox.ui.watch.view.components.cockpit.primitives import (
    _draw_availability_led,
    _draw_centered_label,
)
from rl_fzerox.ui.watch.view.components.cockpit.speed import draw_speed_gauge
from rl_fzerox.ui.watch.view.components.cockpit.style import (
    AVAILABILITY_LED_STYLE,
    COCKPIT_PANEL_STYLE,
    THRUST_COLUMN_STYLE,
)
from rl_fzerox.ui.watch.view.components.cockpit.thrust import _draw_thrust_column
from rl_fzerox.ui.watch.view.components.tokens import _pill_height
from rl_fzerox.ui.watch.view.screen.layout import LAYOUT
from rl_fzerox.ui.watch.view.screen.theme import PALETTE
from rl_fzerox.ui.watch.view.screen.types import (
    ControlViz,
    PygameModule,
    PygameRect,
    PygameSurface,
    ViewerFonts,
)


def _draw_control_viz(
    *,
    pygame: PygameModule,
    screen: PygameSurface,
    fonts: ViewerFonts,
    x: int,
    y: int,
    width: int,
    control_viz: ControlViz,
) -> int:
    wide = width >= COCKPIT_PANEL_STYLE.wide_control_min_width
    panel_height = _control_viz_panel_height(width)
    panel = pygame.Rect(x, y, width, panel_height)
    _draw_cockpit_panel(pygame=pygame, screen=screen, rect=panel)

    header = fonts.small.render("COCKPIT CONTROL", True, PALETTE.text_muted)
    screen.blit(header, (panel.x + 16, panel.y + 9))

    gas_width = 24 if wide else LAYOUT.control_gas_width
    gas_height = 102 if wide else LAYOUT.control_gas_height
    steer_height = 20 if wide else LAYOUT.control_steer_height
    marker_radius = 8 if wide else LAYOUT.control_marker_radius
    aux_gap = 16 if wide else 8
    pitch_width = gas_width
    engine_width = gas_width
    thrust_group_width = (3 * gas_width) + pitch_width + (3 * aux_gap)
    thrust_x = panel.right - (34 if wide else 28) - thrust_group_width
    air_brake_x = thrust_x + gas_width + aux_gap
    pitch_x = air_brake_x + gas_width + aux_gap
    engine_x = pitch_x + pitch_width + aux_gap
    thrust_y = panel.y + (38 if wide else 28)
    led_clearance = 9 if wide else 6

    _draw_centered_label(
        screen=screen,
        font=fonts.small,
        label="THR",
        color=PALETTE.text_muted,
        center_x=thrust_x + (gas_width // 2),
        y=panel.y + (16 if wide else 8),
    )
    _draw_centered_label(
        screen=screen,
        font=fonts.small,
        label="BRK",
        color=PALETTE.text_muted,
        center_x=air_brake_x + (gas_width // 2),
        y=panel.y + (16 if wide else 8),
    )
    _draw_centered_label(
        screen=screen,
        font=fonts.small,
        label="PCH",
        color=PALETTE.text_muted,
        center_x=pitch_x + (pitch_width // 2),
        y=panel.y + (16 if wide else 8),
    )
    _draw_centered_label(
        screen=screen,
        font=fonts.small,
        label="ENG",
        color=PALETTE.text_muted,
        center_x=engine_x + (engine_width // 2),
        y=panel.y + (16 if wide else 8),
    )

    steer_x = panel.x + (20 if wide else 14)
    steer_width = max(190 if wide else 84, thrust_x - steer_x - (28 if wide else 22))
    steer_track = pygame.Rect(
        steer_x,
        panel.y + (58 if wide else 42),
        steer_width,
        steer_height,
    )
    draw_steer_instrument(
        pygame=pygame,
        screen=screen,
        track=steer_track,
        value=control_viz.steer_x,
        marker_radius=marker_radius,
    )

    lean_y = panel.y + (108 if wide else 76)
    lean_width = 30 if wide else 24
    lean_gap = 12 if wide else 6
    boost_radius = 17 if wide else 11
    boost_center_x = steer_x + lean_width + lean_gap + boost_radius
    right_lean_x = boost_center_x + boost_radius + lean_gap
    boost_center_y = lean_y + ((_pill_height(fonts.small) + 2) // 2)
    left_lean_rect = _draw_lean_button(
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
    right_lean_rect = _draw_lean_button(
        pygame=pygame,
        screen=screen,
        font=fonts.small,
        x=right_lean_x,
        y=lean_y,
        width=lean_width,
        direction=1,
        active=control_viz.lean_direction > 0,
    )
    led_radius = AVAILABILITY_LED_STYLE.radius
    lean_led_y = max(left_lean_rect.bottom, right_lean_rect.bottom) + led_clearance + led_radius
    _draw_availability_led(
        pygame=pygame,
        screen=screen,
        center=(left_lean_rect.centerx, lean_led_y),
        available=not control_viz.lean_left_masked,
    )
    _draw_availability_led(
        pygame=pygame,
        screen=screen,
        center=(boost_center_x, boost_center_y + boost_radius + led_clearance + led_radius),
        available=not control_viz.boost_masked,
    )
    _draw_availability_led(
        pygame=pygame,
        screen=screen,
        center=(right_lean_rect.centerx, lean_led_y),
        available=not control_viz.lean_right_masked,
    )
    if wide:
        speed_x = right_lean_rect.right + 18
        speed_width = min(220, max(0, thrust_x - speed_x - 12))
        if speed_width >= 140:
            draw_speed_gauge(
                pygame=pygame,
                screen=screen,
                rect=pygame.Rect(speed_x, lean_y - 7, speed_width, 88),
                speed_kph=control_viz.speed_kph,
                value_font=fonts.body,
            )

    gas_level = max(0.0, min(1.0, control_viz.gas_level))
    thrust_color = (
        THRUST_COLUMN_STYLE.warning_fill
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
        deadzone_threshold=control_viz.thrust_deadzone_threshold,
        full_threshold=control_viz.thrust_full_threshold,
        fill_color=thrust_color,
        width=gas_width,
        height=gas_height,
    )
    air_brake_level = max(0.0, min(1.0, control_viz.air_brake_axis or 0.0))
    air_brake_color = (
        PALETTE.text_muted if control_viz.air_brake_disabled else THRUST_COLUMN_STYLE.warning_fill
    )
    air_brake_rect = pygame.Rect(air_brake_x, thrust_y, gas_width, gas_height)
    _draw_thrust_column(
        pygame=pygame,
        screen=screen,
        x=air_brake_x,
        y=thrust_y,
        level=air_brake_level,
        deadzone_threshold=None,
        full_threshold=None,
        fill_color=air_brake_color,
        width=gas_width,
        height=gas_height,
    )
    pitch_track = pygame.Rect(pitch_x, thrust_y, pitch_width, gas_height)
    draw_pitch_instrument(
        pygame=pygame,
        screen=screen,
        track=pitch_track,
        value=control_viz.pitch_y,
        marker_radius=marker_radius,
    )
    engine_level = control_viz.engine_setting_level
    _draw_thrust_column(
        pygame=pygame,
        screen=screen,
        x=engine_x,
        y=thrust_y,
        level=0.0 if engine_level is None else engine_level,
        deadzone_threshold=None,
        full_threshold=None,
        fill_color=PALETTE.text_warning,
        width=engine_width,
        height=gas_height,
    )
    percent = fonts.body.render(
        f"{round(gas_level * 100):3d}%",
        True,
        thrust_color if gas_level > 0.0 else PALETTE.text_muted,
    )
    percent_x = thrust_x + (gas_width // 2) - (percent.get_width() // 2)
    percent_y = thrust_y + gas_height + 4
    screen.blit(percent, (percent_x, percent_y))
    meter_led_y = percent_y + percent.get_height() + led_clearance + led_radius
    _draw_availability_led(
        pygame=pygame,
        screen=screen,
        center=(air_brake_rect.centerx, meter_led_y),
        available=not (control_viz.air_brake_masked or control_viz.air_brake_disabled),
    )
    _draw_availability_led(
        pygame=pygame,
        screen=screen,
        center=(pitch_track.centerx, meter_led_y),
        available=not control_viz.pitch_masked,
    )
    engine_label = "-" if engine_level is None else f"{round(engine_level * 100):02d}"
    engine_surface = fonts.body.render(engine_label, True, PALETTE.text_warning)
    screen.blit(
        engine_surface,
        (
            engine_x + (engine_width // 2) - (engine_surface.get_width() // 2),
            percent_y,
        ),
    )
    return panel.bottom


def _control_viz_panel_height(width: int) -> int:
    style = COCKPIT_PANEL_STYLE
    if width >= style.wide_control_min_width:
        return style.wide_panel_height
    return style.compact_panel_height


def _draw_cockpit_panel(*, pygame: PygameModule, screen: PygameSurface, rect: PygameRect) -> None:
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
    style = COCKPIT_PANEL_STYLE
    pygame.draw.polygon(screen, style.fill, points)
    for x_offset in range(18, rect.width, 32):
        pygame.draw.line(
            screen,
            style.grid,
            (rect.left + x_offset, rect.top + 1),
            (rect.left + x_offset - 28, rect.bottom - 1),
        )
    pygame.draw.polygon(screen, style.border, points, width=1)
