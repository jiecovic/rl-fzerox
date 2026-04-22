# src/rl_fzerox/ui/watch/input.py
from __future__ import annotations

from dataclasses import dataclass

from fzerox_emulator import (
    JOYPAD_A,
    JOYPAD_B,
    JOYPAD_DOWN,
    JOYPAD_LEFT,
    JOYPAD_RIGHT,
    JOYPAD_SELECT,
    JOYPAD_START,
    JOYPAD_UP,
    ControllerState,
    joypad_mask,
)
from rl_fzerox.ui.watch.view.screen.types import MouseRect


@dataclass(frozen=True)
class ViewerInput:
    """Normalized viewer input state for one polling cycle."""

    quit_requested: bool = False
    toggle_pause: bool = False
    step_once: bool = False
    save_state: bool = False
    force_reset: bool = False
    toggle_deterministic_policy: bool = False
    control_fps_delta: int = 0
    control_state: ControllerState = ControllerState()


def _poll_viewer_input(
    pygame,
    *,
    deterministic_toggle_rect: MouseRect | None = None,
) -> ViewerInput:
    quit_requested = False
    toggle_pause = False
    step_once = False
    save_state = False
    force_reset = False
    toggle_deterministic_policy = False
    control_fps_delta = 0

    mouse_button_down = getattr(pygame, "MOUSEBUTTONDOWN", None)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            quit_requested = True
        elif event.type == mouse_button_down and event.button == 1:
            if _point_in_rect(event.pos, deterministic_toggle_rect):
                toggle_deterministic_policy = True
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_p:
                toggle_pause = True
            elif event.key == pygame.K_n:
                step_once = True
            elif event.key == pygame.K_k:
                save_state = True
            elif event.key == pygame.K_r:
                force_reset = True
            elif event.key == pygame.K_d:
                toggle_deterministic_policy = True
            elif event.key in (pygame.K_PLUS, pygame.K_KP_PLUS):
                control_fps_delta += 1
            elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                control_fps_delta -= 1

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
        force_reset=force_reset,
        toggle_deterministic_policy=toggle_deterministic_policy,
        control_fps_delta=control_fps_delta,
        control_state=ControllerState(
            joypad_mask=joypad_mask(*pressed_buttons),
            left_stick_x=left_stick_x,
        ),
    )


def _point_in_rect(position: object, rect: MouseRect | None) -> bool:
    if rect is None:
        return False
    if not isinstance(position, tuple) or len(position) != 2:
        return False
    x, y = position
    if not isinstance(x, int | float) or not isinstance(y, int | float):
        return False
    rect_x, rect_y, rect_width, rect_height = rect
    return rect_x <= x < rect_x + rect_width and rect_y <= y < rect_y + rect_height
