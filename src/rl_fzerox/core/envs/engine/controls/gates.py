# src/rl_fzerox/core/envs/engine/controls/gates.py
from __future__ import annotations

from fzerox_emulator import ControllerState


def without_joypad_mask(control_state: ControllerState, joypad_mask: int) -> ControllerState:
    if not control_state.joypad_mask & joypad_mask:
        return control_state
    return ControllerState(
        joypad_mask=control_state.joypad_mask & ~joypad_mask,
        left_stick_x=control_state.left_stick_x,
        left_stick_y=control_state.left_stick_y,
        right_stick_x=control_state.right_stick_x,
        right_stick_y=control_state.right_stick_y,
    )


def with_left_stick_y(control_state: ControllerState, left_stick_y: float) -> ControllerState:
    if control_state.left_stick_y == left_stick_y:
        return control_state
    return ControllerState(
        joypad_mask=control_state.joypad_mask,
        left_stick_x=control_state.left_stick_x,
        left_stick_y=left_stick_y,
        right_stick_x=control_state.right_stick_x,
        right_stick_y=control_state.right_stick_y,
    )
