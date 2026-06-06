# src/rl_fzerox/ui/watch/runtime/career_mode/menu.py
"""Map Career Mode semantic menu inputs to emulator controller states."""

from __future__ import annotations

from dataclasses import dataclass

from fzerox_emulator import JOYPAD_BUTTONS, ControllerState, joypad_mask

from rl_fzerox.core.career_mode.runner.menu import MenuInput, RawMenuStep
from rl_fzerox.core.envs.engine.reset.camera import CAMERA_SYNC_CONTROLS


@dataclass(frozen=True, slots=True)
class RawControllerStep:
    """One emulator-controller pulse emitted from a semantic menu step."""

    controller_state: ControllerState
    frames: int
    phase: str

    def __post_init__(self) -> None:
        if self.frames <= 0:
            raise ValueError("frames must be positive")
        if not self.phase:
            raise ValueError("phase must not be empty")


@dataclass(frozen=True, slots=True)
class CareerMenuControls:
    """Controller states for Career Mode menu inputs."""

    neutral: ControllerState = ControllerState()
    start: ControllerState = ControllerState(joypad_mask=joypad_mask(JOYPAD_BUTTONS.start))
    accept: ControllerState = ControllerState(joypad_mask=joypad_mask(JOYPAD_BUTTONS.a))
    a_button: ControllerState = ControllerState(joypad_mask=joypad_mask(JOYPAD_BUTTONS.a))
    cancel: ControllerState = ControllerState(joypad_mask=joypad_mask(JOYPAD_BUTTONS.b))
    up: ControllerState = ControllerState(joypad_mask=joypad_mask(JOYPAD_BUTTONS.up))
    down: ControllerState = ControllerState(joypad_mask=joypad_mask(JOYPAD_BUTTONS.down))
    left: ControllerState = ControllerState(joypad_mask=joypad_mask(JOYPAD_BUTTONS.left))
    right: ControllerState = ControllerState(joypad_mask=joypad_mask(JOYPAD_BUTTONS.right))
    next_camera: ControllerState = CAMERA_SYNC_CONTROLS.next_camera


MENU_CONTROLS = CareerMenuControls()


_MENU_INPUT_STATES = {
    MenuInput.NEUTRAL: MENU_CONTROLS.neutral,
    MenuInput.ACCEPT: MENU_CONTROLS.accept,
    MenuInput.A_BUTTON: MENU_CONTROLS.a_button,
    MenuInput.CANCEL: MENU_CONTROLS.cancel,
    MenuInput.START: MENU_CONTROLS.start,
    MenuInput.UP: MENU_CONTROLS.up,
    MenuInput.DOWN: MENU_CONTROLS.down,
    MenuInput.LEFT: MENU_CONTROLS.left,
    MenuInput.RIGHT: MENU_CONTROLS.right,
    MenuInput.NEXT_CAMERA: MENU_CONTROLS.next_camera,
}


def controller_step_from_menu_step(step: RawMenuStep) -> RawControllerStep:
    return RawControllerStep(
        controller_state=_MENU_INPUT_STATES[step.menu_input],
        frames=step.frames,
        phase=step.phase,
    )


def neutral_controller_step(*, phase: str) -> RawControllerStep:
    return RawControllerStep(
        controller_state=MENU_CONTROLS.neutral,
        frames=1,
        phase=phase,
    )
