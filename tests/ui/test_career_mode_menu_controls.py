# tests/ui/test_career_mode_menu_controls.py

from fzerox_emulator.control import ControllerState
from fzerox_emulator import JOYPAD_BUTTONS, joypad_mask

from rl_fzerox.core.career_mode.runner.menu import MenuInput, RawMenuStep
from rl_fzerox.ui.watch.runtime.career_mode.menu import controller_step_from_menu_step


def test_career_menu_accept_uses_a_and_start_stays_explicit() -> None:
    accept = controller_step_from_menu_step(RawMenuStep(MenuInput.ACCEPT, frames=1, phase="accept"))
    a_button = controller_step_from_menu_step(RawMenuStep(MenuInput.A_BUTTON, frames=1, phase="a"))
    start = controller_step_from_menu_step(RawMenuStep(MenuInput.START, frames=1, phase="start"))

    assert accept.controller_state.joypad_mask == joypad_mask(JOYPAD_BUTTONS.a)
    assert a_button.controller_state.joypad_mask == joypad_mask(JOYPAD_BUTTONS.a)
    assert start.controller_state.joypad_mask == joypad_mask(JOYPAD_BUTTONS.start)
    assert accept.controller_state.joypad_mask == a_button.controller_state.joypad_mask
    assert accept.controller_state.joypad_mask != start.controller_state.joypad_mask


def test_career_menu_navigation_inputs_do_not_move_sticks() -> None:
    for menu_input in MenuInput:
        if menu_input == MenuInput.NEXT_CAMERA:
            continue
        step = controller_step_from_menu_step(
            RawMenuStep(menu_input, frames=1, phase=menu_input.value)
        )

        _assert_sticks_neutral(step.controller_state)


def test_career_camera_sync_is_the_only_c_axis_menu_control() -> None:
    step = controller_step_from_menu_step(
        RawMenuStep(MenuInput.NEXT_CAMERA, frames=1, phase="camera")
    )

    assert step.controller_state.joypad_mask == 0
    assert step.controller_state.left_stick_x == 0.0
    assert step.controller_state.left_stick_y == 0.0
    assert step.controller_state.right_stick_x != 0.0
    assert step.controller_state.right_stick_y == 0.0


def _assert_sticks_neutral(controller_state: ControllerState) -> None:
    assert controller_state.left_stick_x == 0.0
    assert controller_state.left_stick_y == 0.0
    assert controller_state.right_stick_x == 0.0
    assert controller_state.right_stick_y == 0.0
