# src/rl_fzerox/ui/watch/hud/viz.py
from __future__ import annotations

from fzerox_emulator import ControllerState
from rl_fzerox.core.envs.actions import (
    BOOST_MASK,
    BRAKE_MASK,
    DRIFT_LEFT_MASK,
    DRIFT_RIGHT_MASK,
    THROTTLE_MASK,
)
from rl_fzerox.ui.watch.layout import LAYOUT, PALETTE, ControlViz, FlagToken, FlagViz, ViewerFonts


def _control_viz(control_state: ControllerState) -> ControlViz:
    joypad_mask = control_state.joypad_mask
    drive_level = 1 if joypad_mask & THROTTLE_MASK else -1 if joypad_mask & BRAKE_MASK else 0
    return ControlViz(
        steer_x=max(-1.0, min(1.0, control_state.left_stick_x)),
        drive_level=drive_level,
        boost_pressed=bool(joypad_mask & BOOST_MASK),
        drift_direction=(
            -1 if joypad_mask & DRIFT_LEFT_MASK else 1 if joypad_mask & DRIFT_RIGHT_MASK else 0
        ),
    )


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
