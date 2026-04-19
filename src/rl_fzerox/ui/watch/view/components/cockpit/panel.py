# src/rl_fzerox/ui/watch/view/components/cockpit/panel.py
from __future__ import annotations

from rl_fzerox.ui.watch.view.components.cockpit.buttons import (
    _draw_boost_button,
    _draw_lean_button,
)
from rl_fzerox.ui.watch.view.components.cockpit.primitives import _draw_centered_label
from rl_fzerox.ui.watch.view.components.cockpit.steer import _draw_steer_instrument
from rl_fzerox.ui.watch.view.components.cockpit.style import (
    COCKPIT_GRID,
    COCKPIT_PANEL_BORDER,
    COCKPIT_PANEL_FILL,
    THRUST_WARNING_FILL,
)
from rl_fzerox.ui.watch.view.components.cockpit.thrust import _draw_thrust_column
from rl_fzerox.ui.watch.view.components.tokens import _pill_height
from rl_fzerox.ui.watch.view.screen.layout import LAYOUT
from rl_fzerox.ui.watch.view.screen.theme import PALETTE
from rl_fzerox.ui.watch.view.screen.types import ControlViz, ViewerFonts

_WIDE_CONTROL_MIN_WIDTH = 420


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
    wide = width >= _WIDE_CONTROL_MIN_WIDTH
    panel_height = _control_viz_panel_height(width)
    panel = pygame.Rect(x, y, width, panel_height)
    _draw_cockpit_panel(pygame=pygame, screen=screen, rect=panel)

    header = fonts.small.render("COCKPIT CONTROL", True, PALETTE.text_muted)
    screen.blit(header, (panel.x + 16, panel.y + 9))

    dual_gas_levers = control_viz.air_brake_axis is not None
    gas_width = 24 if wide else LAYOUT.control_gas_width
    gas_height = 102 if wide else LAYOUT.control_gas_height
    steer_height = 20 if wide else LAYOUT.control_steer_height
    marker_radius = 8 if wide else LAYOUT.control_marker_radius
    thrust_gap = 28 if wide and dual_gas_levers else 22 if dual_gas_levers else 0
    thrust_group_width = (2 * gas_width) + thrust_gap if dual_gas_levers else gas_width
    thrust_x = panel.right - (34 if wide else 28) - thrust_group_width
    air_brake_x = thrust_x + gas_width + thrust_gap
    thrust_y = panel.y + (38 if wide else 28)

    _draw_centered_label(
        screen=screen,
        font=fonts.small,
        label="THR" if dual_gas_levers else "THRUST",
        color=PALETTE.text_muted,
        center_x=thrust_x + (gas_width // 2),
        y=panel.y + (16 if wide else 8),
    )
    if dual_gas_levers:
        _draw_centered_label(
            screen=screen,
            font=fonts.small,
            label="BRK",
            color=PALETTE.text_muted,
            center_x=air_brake_x + (gas_width // 2),
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
    _draw_steer_instrument(
        pygame=pygame,
        screen=screen,
        track=steer_track,
        value=control_viz.steer_x,
        marker_radius=marker_radius,
    )

    lean_y = panel.y + (108 if wide else 76)
    lean_width = 30 if wide else 24 if dual_gas_levers else 26
    lean_gap = 12 if wide else 6 if dual_gas_levers else 8
    boost_radius = 17 if wide else 11 if dual_gas_levers else 13
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
        width=gas_width,
        height=gas_height,
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
            width=gas_width,
            height=gas_height,
        )

    percent = fonts.body.render(
        f"{round(gas_level * 100):3d}%",
        True,
        thrust_color if gas_level > 0.0 else PALETTE.text_muted,
    )
    percent_x = thrust_x + (thrust_group_width // 2) - (percent.get_width() // 2)
    screen.blit(percent, (percent_x, thrust_y + gas_height + 4))
    return panel.bottom


def _control_viz_panel_height(width: int) -> int:
    return 160 if width >= _WIDE_CONTROL_MIN_WIDTH else 118


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
