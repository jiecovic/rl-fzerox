# src/rl_fzerox/ui/watch/hud/viz.py
from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from fzerox_emulator import ControllerState
from rl_fzerox.core.envs.actions import (
    ACCELERATE_MASK,
    AIR_BRAKE_MASK,
    BOOST_MASK,
    DRIFT_LEFT_MASK,
    DRIFT_RIGHT_MASK,
    ActionValue,
)
from rl_fzerox.core.hybrid_action import HYBRID_CONTINUOUS_ACTION_KEY
from rl_fzerox.ui.watch.layout import LAYOUT, PALETTE, ControlViz, FlagToken, FlagViz, ViewerFonts


def _control_viz(
    control_state: ControllerState,
    *,
    policy_action: ActionValue | None = None,
    continuous_drive_mode: str = "threshold",
    continuous_drive_deadzone: float = 0.2,
) -> ControlViz:
    joypad_mask = control_state.joypad_mask
    drive_level = 1 if joypad_mask & ACCELERATE_MASK else -1 if joypad_mask & AIR_BRAKE_MASK else 0
    drive_axis, air_brake_axis = _continuous_drive_axes(
        policy_action,
        continuous_drive_mode=continuous_drive_mode,
        continuous_drive_deadzone=continuous_drive_deadzone,
    )
    return ControlViz(
        steer_x=max(-1.0, min(1.0, control_state.left_stick_x)),
        drive_level=drive_level,
        drive_axis=drive_axis,
        air_brake_axis=air_brake_axis,
        drive_axis_mode=(
            "accelerate"
            if drive_axis is not None and continuous_drive_mode == "pwm"
            else "signed"
        ),
        boost_pressed=bool(joypad_mask & BOOST_MASK),
        drift_direction=(
            -1 if joypad_mask & DRIFT_LEFT_MASK else 1 if joypad_mask & DRIFT_RIGHT_MASK else 0
        ),
    )


def _continuous_drive_axes(
    policy_action: ActionValue | None,
    *,
    continuous_drive_mode: str,
    continuous_drive_deadzone: float,
) -> tuple[float | None, float | None]:
    """Return HUD accelerate/air-brake values when the policy action exposes those axes."""

    if policy_action is None:
        return None, None
    source = (
        policy_action.get(HYBRID_CONTINUOUS_ACTION_KEY)
        if isinstance(policy_action, Mapping)
        else policy_action
    )
    action = np.asarray(source)
    if not np.issubdtype(action.dtype, np.floating):
        return None, None
    values = action.reshape(-1)
    if values.size < 2:
        return None, None
    drive = float(values[1])
    if not np.isfinite(drive):
        return None, None
    drive = max(-1.0, min(1.0, drive))
    air_brake = _continuous_air_brake_axis(
        values,
        continuous_drive_deadzone=continuous_drive_deadzone,
    )
    if continuous_drive_mode == "pwm":
        return _continuous_drive_accelerate_level(
            drive,
            deadzone=continuous_drive_deadzone,
        ), air_brake
    return drive, air_brake


def _continuous_air_brake_axis(
    values: np.ndarray,
    *,
    continuous_drive_deadzone: float,
) -> float | None:
    if values.size < 3:
        return None
    air_brake = float(values[2])
    if not np.isfinite(air_brake):
        return None
    air_brake = max(-1.0, min(1.0, air_brake))
    return _continuous_positive_button_level(
        air_brake,
        deadzone=continuous_drive_deadzone,
    )


def _continuous_drive_accelerate_level(drive: float, *, deadzone: float) -> float:
    duty = (drive + 1.0) * 0.5
    if duty <= deadzone:
        return 0.0
    return (duty - deadzone) / (1.0 - deadzone)


def _continuous_positive_button_level(value: float, *, deadzone: float) -> float:
    if value <= deadzone:
        return 0.0
    return (value - deadzone) / (1.0 - deadzone)


def _control_viz_height(fonts: ViewerFonts) -> int:
    small_height = fonts.small.render("Drive", True, PALETTE.text_primary).get_height()
    return (
        small_height
        + LAYOUT.control_track_gap
        + LAYOUT.control_drive_height
        + LAYOUT.control_caption_gap
        + small_height
        + LAYOUT.control_boost_gap
        + small_height
        + (2 * LAYOUT.flag_token_pad_y)
    )


_FLAG_DISPLAY_LABELS = {
    "reverse_detected": "reverse",
    "low_speed_detected": "slow",
    "can_boost": "can boost",
    "boost_active": "boost",
    "airborne": "airborne",
    "collision_recoil": "recoil",
    "spinning_out": "spin",
    "crashed": "crash",
    "falling_off_track": "off-track",
    "energy_depleted": "depleted",
    "retired": "retired",
    "finished": "finish",
}

_FLAG_ROWS = (
    ("reverse_detected", "low_speed_detected"),
    ("can_boost", "boost_active", "airborne"),
    ("collision_recoil", "spinning_out", "crashed", "falling_off_track"),
    ("energy_depleted", "retired", "finished"),
)


def _flag_viz(
    state_labels: tuple[str, ...],
    *,
    boost_active: bool,
    reverse_detected: bool,
    low_speed_detected: bool,
    energy_depleted: bool,
) -> FlagViz:
    active_flags = set(state_labels)
    if boost_active:
        active_flags.add("boost_active")
    if reverse_detected:
        active_flags.add("reverse_detected")
    if low_speed_detected:
        active_flags.add("low_speed_detected")
    if energy_depleted:
        active_flags.add("energy_depleted")
    known_flags = {flag_label for row in _FLAG_ROWS for flag_label in row}
    active_flags.intersection_update(known_flags)
    return FlagViz(
        rows=tuple(
            tuple(
                FlagToken(
                    label=_FLAG_DISPLAY_LABELS.get(flag_label, flag_label.replace("_", " ")),
                    active=flag_label in active_flags,
                )
                for flag_label in row
            )
            for row in _FLAG_ROWS
        )
    )


def _flag_viz_height(fonts: ViewerFonts, flag_viz: FlagViz) -> int:
    row_height = fonts.small.render("flags", True, PALETTE.text_primary).get_height()
    pill_height = row_height + (2 * LAYOUT.flag_token_pad_y)
    return row_height + LAYOUT.line_gap + (len(flag_viz.rows) * (pill_height + LAYOUT.line_gap))


def _wrap_text(font, text: str, max_width: int) -> list[str]:
    normalized = " ".join(text.split())
    if not normalized:
        return [""]
    if font.render(normalized, True, PALETTE.text_primary).get_width() <= max_width:
        return [normalized]

    wrapped: list[str] = []
    current = ""
    for token in normalized.split(" "):
        candidate = token if not current else f"{current} {token}"
        if current and font.render(candidate, True, PALETTE.text_primary).get_width() > max_width:
            wrapped.append(current)
            current = token
        else:
            current = candidate
    if current:
        wrapped.append(current)
    return wrapped
