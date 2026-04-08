# src/rl_fzerox/ui/watch/input.py
from __future__ import annotations

from dataclasses import dataclass

from rl_fzerox._native import (
    JOYPAD_A,
    JOYPAD_B,
    JOYPAD_DOWN,
    JOYPAD_LEFT,
    JOYPAD_RIGHT,
    JOYPAD_SELECT,
    JOYPAD_START,
    JOYPAD_UP,
    joypad_mask,
)
from rl_fzerox.core.emulator.control import ControllerState


@dataclass(frozen=True)
class ViewerInput:
    """Normalized viewer input state for one polling cycle."""

    quit_requested: bool = False
    toggle_pause: bool = False
    step_once: bool = False
    save_state: bool = False
    control_state: ControllerState = ControllerState()


def _poll_viewer_input(pygame) -> ViewerInput:
    quit_requested = False
    toggle_pause = False
    step_once = False
    save_state = False

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            quit_requested = True
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_p:
                toggle_pause = True
            elif event.key == pygame.K_n:
                step_once = True
            elif event.key == pygame.K_k:
                save_state = True

    keys = pygame.key.get_pressed()
    pressed_buttons: list[int] = []
    if keys[pygame.K_UP]:
        pressed_buttons.append(JOYPAD_UP)
    if keys[pygame.K_DOWN]:
        pressed_buttons.append(JOYPAD_DOWN)
    if keys[pygame.K_LEFT]:
        pressed_buttons.append(JOYPAD_LEFT)
    if keys[pygame.K_RIGHT]:
        pressed_buttons.append(JOYPAD_RIGHT)
    if keys[pygame.K_x]:
        pressed_buttons.append(JOYPAD_A)
    if keys[pygame.K_z]:
        pressed_buttons.append(JOYPAD_B)
    if keys[pygame.K_RETURN]:
        pressed_buttons.append(JOYPAD_START)
    if keys[pygame.K_BACKSPACE]:
        pressed_buttons.append(JOYPAD_SELECT)

    left_stick_x = 0.0
    if keys[pygame.K_LEFT] and not keys[pygame.K_RIGHT]:
        left_stick_x = -1.0
    elif keys[pygame.K_RIGHT] and not keys[pygame.K_LEFT]:
        left_stick_x = 1.0

    return ViewerInput(
        quit_requested=quit_requested,
        toggle_pause=toggle_pause,
        step_once=step_once,
        save_state=save_state,
        control_state=ControllerState(
            joypad_mask=joypad_mask(*pressed_buttons),
            left_stick_x=left_stick_x,
        ),
    )
