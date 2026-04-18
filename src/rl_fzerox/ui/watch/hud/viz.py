# src/rl_fzerox/ui/watch/hud/viz.py
from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from fzerox_emulator import ControllerState
from fzerox_emulator.arrays import ContinuousAction
from rl_fzerox.core.domain.hybrid_action import HYBRID_CONTINUOUS_ACTION_KEY
from rl_fzerox.core.envs.actions import (
    BOOST_MASK,
    LEAN_LEFT_MASK,
    LEAN_RIGHT_MASK,
    ActionValue,
)
from rl_fzerox.ui.watch.layout import LAYOUT, PALETTE, ControlViz, FlagToken, FlagViz, ViewerFonts


def _control_viz(
    control_state: ControllerState,
    *,
    gas_level: float,
    thrust_warning_threshold: float | None = None,
    boost_active: bool = False,
    boost_lamp_level: float = 0.0,
    policy_action: ActionValue | None = None,
    continuous_drive_deadzone: float = 0.2,
    continuous_air_brake_mode: str = "always",
    continuous_air_brake_disabled: bool = False,
) -> ControlViz:
    joypad_mask = control_state.joypad_mask
    air_brake_axis = _continuous_air_brake_axis_from_action(
        policy_action,
        continuous_drive_deadzone=continuous_drive_deadzone,
        continuous_air_brake_enabled=continuous_air_brake_mode != "off",
    )
    boost_pressed = bool(joypad_mask & BOOST_MASK)
    normalized_boost_lamp_level = max(
        0.0,
        min(
            1.0,
            max(
                boost_lamp_level,
                1.0 if boost_pressed else 0.0,
                0.55 if boost_active else 0.0,
            ),
        ),
    )
    return ControlViz(
        steer_x=max(-1.0, min(1.0, control_state.left_stick_x)),
        gas_level=max(0.0, min(1.0, gas_level)),
        thrust_warning_threshold=_normalize_optional_level(thrust_warning_threshold),
        air_brake_axis=air_brake_axis,
        air_brake_disabled=continuous_air_brake_disabled and air_brake_axis is not None,
        boost_pressed=boost_pressed,
        boost_active=boost_active,
        boost_lamp_level=normalized_boost_lamp_level,
        lean_direction=(
            -1 if joypad_mask & LEAN_LEFT_MASK else 1 if joypad_mask & LEAN_RIGHT_MASK else 0
        ),
    )


def _continuous_air_brake_axis_from_action(
    policy_action: ActionValue | None,
    *,
    continuous_drive_deadzone: float,
    continuous_air_brake_enabled: bool,
) -> float | None:
    """Return HUD air-brake value when the policy action exposes that axis."""

    if policy_action is None:
        return None
    source = (
        policy_action.get(HYBRID_CONTINUOUS_ACTION_KEY)
        if isinstance(policy_action, Mapping)
        else policy_action
    )
    action = np.asarray(source)
    if not np.issubdtype(action.dtype, np.floating):
        return None
    values = action.reshape(-1)
    if values.size < 3 or not continuous_air_brake_enabled:
        return None
    return _continuous_air_brake_axis(
        values,
        continuous_drive_deadzone=continuous_drive_deadzone,
    )


def _continuous_air_brake_axis(
    values: ContinuousAction,
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


def _continuous_positive_button_level(value: float, *, deadzone: float) -> float:
    if value <= deadzone:
        return 0.0
    return (value - deadzone) / (1.0 - deadzone)


def _normalize_optional_level(value: float | None) -> float | None:
    if value is None:
        return None
    return max(0.0, min(1.0, float(value)))


def _control_viz_height(fonts: ViewerFonts) -> int:
    small_height = fonts.small.render("Thrust", True, PALETTE.text_primary).get_height()
    return (
        small_height
        + LAYOUT.control_track_gap
        + LAYOUT.control_gas_height
        + (2 * LAYOUT.control_caption_gap)
        + (2 * small_height)
        + (2 * LAYOUT.flag_token_pad_y)
    )


_FLAG_DISPLAY_LABELS = {
    "reverse_detected": "reverse",
    "low_speed_detected": "slow",
    "can_boost": "can boost",
    "boost_active": "boost",
    "energy_loss": "energy loss",
    "damage_taken": "damage taken",
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
    ("energy_loss", "damage_taken"),
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
    energy_loss_detected: bool = False,
    damage_taken_detected: bool = False,
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
    if energy_loss_detected:
        active_flags.add("energy_loss")
    if damage_taken_detected:
        active_flags.add("damage_taken")
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
