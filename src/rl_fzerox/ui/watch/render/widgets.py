# src/rl_fzerox/ui/watch/render/widgets.py
from __future__ import annotations

from rl_fzerox.ui.watch.layout import LAYOUT, PALETTE, ControlViz, FlagViz, ViewerFonts


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
    steer_label = fonts.small.render("Steer", True, PALETTE.text_muted)
    dual_drive_levers = control_viz.drive_axis_mode == "gas" and control_viz.brake_axis is not None
    drive_group_width = (
        (2 * LAYOUT.control_drive_width) + LAYOUT.control_drive_pair_gap
        if dual_drive_levers
        else LAYOUT.control_drive_width
    )
    drive_x = x + width - drive_group_width - LAYOUT.control_drive_offset_x
    brake_x = drive_x + LAYOUT.control_drive_width + LAYOUT.control_drive_pair_gap
    left_widget_width = max(
        48,
        width
        - drive_group_width
        - LAYOUT.control_drive_offset_x
        - LAYOUT.control_widget_gap,
    )
    left_drift_width = _pill_width(fonts.small, "drift")
    right_drift_width = _pill_width(fonts.small, "drift")
    max_steer_width = max(
        48,
        left_widget_width
        - left_drift_width
        - right_drift_width
        - (2 * LAYOUT.control_side_pill_gap),
    )
    steer_width = min(LAYOUT.control_steer_width, max_steer_width)
    steer_group_width = (
        left_drift_width
        + LAYOUT.control_side_pill_gap
        + steer_width
        + LAYOUT.control_side_pill_gap
        + right_drift_width
    )
    steer_group_x = x + max(0, (left_widget_width - steer_group_width) // 2)
    left_drift_x = steer_group_x
    steer_x = left_drift_x + left_drift_width + LAYOUT.control_side_pill_gap
    right_drift_x = steer_x + steer_width + LAYOUT.control_side_pill_gap

    screen.blit(steer_label, (x, y))
    if dual_drive_levers:
        _draw_centered_label(
            screen=screen,
            font=fonts.small,
            label="Gas",
            color=PALETTE.text_muted,
            center_x=drive_x + (LAYOUT.control_drive_width // 2),
            y=y,
        )
        _draw_centered_label(
            screen=screen,
            font=fonts.small,
            label="Brk",
            color=PALETTE.text_muted,
            center_x=brake_x + (LAYOUT.control_drive_width // 2),
            y=y,
        )
    else:
        drive_label_text = "Gas" if control_viz.drive_axis_mode == "gas" else "Drive"
        _draw_centered_label(
            screen=screen,
            font=fonts.small,
            label=drive_label_text,
            color=PALETTE.text_muted,
            center_x=drive_x + (LAYOUT.control_drive_width // 2),
            y=y,
        )
    y += steer_label.get_height() + LAYOUT.control_track_gap

    drive_y = y
    steer_y = drive_y + (LAYOUT.control_drive_height - LAYOUT.control_steer_height) // 2
    steer_mid_y = steer_y + LAYOUT.control_steer_height // 2
    steer_mid_x = steer_x + steer_width // 2
    drift_pill_y = steer_mid_y - (_pill_height(fonts.small) // 2)

    _draw_pill(
        pygame=pygame,
        screen=screen,
        font=fonts.small,
        x=left_drift_x,
        y=drift_pill_y,
        label="drift",
        active=control_viz.drift_direction < 0,
        active_text_color=PALETTE.text_primary,
        active_fill_color=PALETTE.flag_active_background,
        active_border_color=PALETTE.flag_active_border,
    )
    _draw_pill(
        pygame=pygame,
        screen=screen,
        font=fonts.small,
        x=right_drift_x,
        y=drift_pill_y,
        label="drift",
        active=control_viz.drift_direction > 0,
        active_text_color=PALETTE.text_primary,
        active_fill_color=PALETTE.flag_active_background,
        active_border_color=PALETTE.flag_active_border,
    )

    steer_track = pygame.Rect(
        steer_x,
        steer_y,
        steer_width,
        LAYOUT.control_steer_height,
    )
    pygame.draw.rect(
        screen,
        PALETTE.control_track,
        steer_track,
        border_radius=LAYOUT.control_steer_height // 2,
    )

    steer_extent = round((steer_width // 2) * min(1.0, abs(control_viz.steer_x)))
    if steer_extent > 0:
        steer_fill = pygame.Rect(
            steer_mid_x - steer_extent if control_viz.steer_x < 0 else steer_mid_x,
            steer_y,
            steer_extent,
            LAYOUT.control_steer_height,
        )
        pygame.draw.rect(
            screen,
            PALETTE.text_primary,
            steer_fill,
            border_radius=LAYOUT.control_steer_height // 2,
        )

    steer_knob_x = steer_mid_x + round((steer_width // 2) * control_viz.steer_x)
    _draw_round_marker(
        pygame=pygame,
        screen=screen,
        color=PALETTE.control_knob,
        center=(steer_knob_x, steer_mid_y),
        radius=LAYOUT.control_marker_radius,
        outline_color=PALETTE.control_knob_outline,
    )

    drive_group_center_x = drive_x + drive_group_width // 2
    drive_mid_y = drive_y + LAYOUT.control_drive_height // 2
    if dual_drive_levers and control_viz.drive_axis is not None:
        gas_level = max(0.0, min(1.0, control_viz.drive_axis))
        brake_level = max(0.0, min(1.0, control_viz.brake_axis or 0.0))
        _draw_unipolar_drive_lever(
            pygame=pygame,
            screen=screen,
            x=drive_x,
            y=drive_y,
            level=gas_level,
            fill_color=PALETTE.text_accent,
        )
        _draw_unipolar_drive_lever(
            pygame=pygame,
            screen=screen,
            x=brake_x,
            y=drive_y,
            level=brake_level,
            fill_color=PALETTE.text_warning,
        )
    elif control_viz.drive_axis_mode == "gas" and control_viz.drive_axis is not None:
        gas_level = max(0.0, min(1.0, control_viz.drive_axis))
        _draw_unipolar_drive_lever(
            pygame=pygame,
            screen=screen,
            x=drive_x,
            y=drive_y,
            level=gas_level,
            fill_color=PALETTE.text_accent,
        )
    else:
        drive_value = (
            control_viz.drive_axis
            if control_viz.drive_axis is not None
            else float(control_viz.drive_level)
        )
        drive_value = max(-1.0, min(1.0, drive_value))
        pygame.draw.rect(
            screen,
            PALETTE.control_track,
            pygame.Rect(
                drive_x,
                drive_y,
                LAYOUT.control_drive_width,
                LAYOUT.control_drive_height,
            ),
            border_radius=LAYOUT.control_drive_width // 2,
        )
        drive_extent = (LAYOUT.control_drive_height // 2) - LAYOUT.control_marker_radius
        drive_knob_y = drive_mid_y - round(drive_extent * drive_value)
        drive_knob_y = max(
            drive_y + LAYOUT.control_marker_radius,
            min(drive_y + LAYOUT.control_drive_height - LAYOUT.control_marker_radius, drive_knob_y),
        )
        if drive_knob_y != drive_mid_y:
            drive_fill = pygame.Rect(
                drive_x,
                min(drive_mid_y, drive_knob_y),
                LAYOUT.control_drive_width,
                abs(drive_knob_y - drive_mid_y),
            )
            pygame.draw.rect(
                screen,
                PALETTE.text_accent if drive_value > 0.0 else PALETTE.text_warning,
                drive_fill,
                border_radius=LAYOUT.control_drive_width // 2,
            )
        _draw_round_marker(
            pygame=pygame,
            screen=screen,
            color=PALETTE.control_knob if drive_value != 0.0 else PALETTE.control_coast,
            center=(drive_x + LAYOUT.control_drive_width // 2, drive_knob_y),
            radius=LAYOUT.control_marker_radius,
            outline_color=PALETTE.control_knob_outline,
        )

    y += LAYOUT.control_drive_height + LAYOUT.control_caption_gap
    if control_viz.drive_axis_mode == "gas" and control_viz.drive_axis is not None:
        gas_level = max(0.0, min(1.0, control_viz.drive_axis))
        if control_viz.brake_axis is None:
            mode = f"{round(gas_level * 100):3d}%"
            mode_color = PALETTE.text_accent if gas_level > 0.0 else PALETTE.text_muted
        else:
            brake_level = max(0.0, min(1.0, control_viz.brake_axis))
            mode = f"{round(gas_level * 100):3d}% {round(brake_level * 100):3d}%"
            mode_color = (
                PALETTE.text_warning
                if brake_level > 0.0
                else PALETTE.text_accent
                if gas_level > 0.0
                else PALETTE.text_muted
            )
    elif control_viz.drive_level > 0:
        mode = "throttle"
        mode_color = PALETTE.text_accent
    elif control_viz.drive_level < 0:
        mode = "brake"
        mode_color = PALETTE.text_warning
    else:
        mode = "coast"
        mode_color = PALETTE.text_muted
    mode_surface = fonts.body.render(mode, True, mode_color)
    mode_x = drive_group_center_x - (mode_surface.get_width() // 2)
    screen.blit(mode_surface, (mode_x, y))
    y += mode_surface.get_height() + LAYOUT.control_boost_gap
    boost_x = drive_group_center_x - (_pill_width(fonts.small, "boost") // 2)
    _draw_pill(
        pygame=pygame,
        screen=screen,
        font=fonts.small,
        x=boost_x,
        y=y,
        label="boost",
        active=control_viz.boost_pressed,
        active_text_color=PALETTE.text_primary,
        active_fill_color=(88, 66, 28),
        active_border_color=PALETTE.text_warning,
    )
    return y + _pill_height(fonts.small)


def _draw_centered_label(*, screen, font, label: str, color, center_x: int, y: int) -> None:
    surface = font.render(label, True, color)
    screen.blit(surface, (center_x - (surface.get_width() // 2), y))


def _draw_unipolar_drive_lever(
    *,
    pygame,
    screen,
    x: int,
    y: int,
    level: float,
    fill_color,
) -> None:
    level = max(0.0, min(1.0, level))
    track = pygame.Rect(
        x,
        y,
        LAYOUT.control_drive_width,
        LAYOUT.control_drive_height,
    )
    pygame.draw.rect(
        screen,
        PALETTE.control_track,
        track,
        border_radius=LAYOUT.control_drive_width // 2,
    )
    extent = LAYOUT.control_drive_height - (2 * LAYOUT.control_marker_radius)
    knob_y = (
        y
        + LAYOUT.control_drive_height
        - LAYOUT.control_marker_radius
        - round(extent * level)
    )
    fill_height = y + LAYOUT.control_drive_height - knob_y
    if fill_height > 0:
        fill = pygame.Rect(
            x,
            knob_y,
            LAYOUT.control_drive_width,
            fill_height,
        )
        pygame.draw.rect(
            screen,
            fill_color,
            fill,
            border_radius=LAYOUT.control_drive_width // 2,
        )
    _draw_round_marker(
        pygame=pygame,
        screen=screen,
        color=PALETTE.control_knob if level > 0.0 else PALETTE.control_coast,
        center=(x + LAYOUT.control_drive_width // 2, knob_y),
        radius=LAYOUT.control_marker_radius,
        outline_color=PALETTE.control_knob_outline,
    )


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
