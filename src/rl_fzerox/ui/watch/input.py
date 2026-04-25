# src/rl_fzerox/ui/watch/input.py
from __future__ import annotations

from dataclasses import dataclass

from fzerox_emulator import (
    JOYPAD_SELECT,
    JOYPAD_START,
    ControllerState,
    joypad_mask,
)
from rl_fzerox.core.envs.actions import (
    ACCELERATE_MASK,
    AIR_BRAKE_MASK,
    BOOST_MASK,
    LEAN_LEFT_MASK,
    LEAN_RIGHT_MASK,
)
from rl_fzerox.ui.watch.view.screen.types import MouseRect, PygameModule


@dataclass(frozen=True)
class ViewerInput:
    """Normalized viewer input state for one polling cycle."""

    quit_requested: bool = False
    toggle_pause: bool = False
    step_once: bool = False
    save_state: bool = False
    force_reset: bool = False
    toggle_deterministic_policy: bool = False
    toggle_manual_control: bool = False
    control_fps_delta: int = 0
    panel_tab_delta: int = 0
    panel_tab_index: int | None = None
    control_state: ControllerState = ControllerState()


def _poll_viewer_input(
    pygame: PygameModule,
    *,
    deterministic_toggle_rect: MouseRect | None = None,
    panel_tab_rects: tuple[MouseRect | None, ...] = (),
) -> ViewerInput:
    quit_requested = False
    toggle_pause = False
    step_once = False
    save_state = False
    force_reset = False
    toggle_deterministic_policy = False
    toggle_manual_control = False
    control_fps_delta = 0
    panel_tab_delta = 0
    panel_tab_index = None

    mouse_button_down = getattr(pygame, "MOUSEBUTTONDOWN", None)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            quit_requested = True
        elif event.type == mouse_button_down and event.button == 1:
            if _point_in_rect(event.pos, deterministic_toggle_rect):
                toggle_deterministic_policy = True
            else:
                selected_tab = _clicked_panel_tab_index(event.pos, panel_tab_rects)
                if selected_tab is not None:
                    panel_tab_index = selected_tab
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                quit_requested = True
            elif event.key == pygame.K_TAB:
                panel_tab_delta += 1
            elif event.key == pygame.K_1:
                panel_tab_index = 0
            elif event.key == pygame.K_2:
                panel_tab_index = 1
            elif event.key == pygame.K_3:
                panel_tab_index = 2
            elif event.key == pygame.K_4:
                panel_tab_index = 3
            elif event.key == pygame.K_5:
                panel_tab_index = 4
            elif event.key == pygame.K_6:
                panel_tab_index = 5
            elif event.key == pygame.K_p:
                toggle_pause = True
            elif event.key == pygame.K_n:
                step_once = True
            elif event.key == pygame.K_k:
                save_state = True
            elif event.key == pygame.K_r:
                force_reset = True
            elif event.key == pygame.K_d:
                toggle_deterministic_policy = True
            elif event.key == pygame.K_m:
                toggle_manual_control = True
            elif event.key in (pygame.K_PLUS, pygame.K_KP_PLUS):
                control_fps_delta += 1
            elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                control_fps_delta -= 1

    keys = pygame.key.get_pressed()
    manual_mask = 0
    if keys[pygame.K_z]:
        manual_mask |= ACCELERATE_MASK
    if keys[pygame.K_x]:
        manual_mask |= AIR_BRAKE_MASK
    if keys[pygame.K_SPACE]:
        manual_mask |= BOOST_MASK
    if keys[pygame.K_a]:
        manual_mask |= LEAN_LEFT_MASK
    if keys[pygame.K_s]:
        manual_mask |= LEAN_RIGHT_MASK
    pressed_buttons: list[int] = []
    if keys[pygame.K_RETURN]:
        pressed_buttons.append(JOYPAD_START)
    if keys[pygame.K_BACKSPACE]:
        pressed_buttons.append(JOYPAD_SELECT)
    manual_mask |= joypad_mask(*pressed_buttons)

    left_stick_x = 0.0
    if keys[pygame.K_LEFT] and not keys[pygame.K_RIGHT]:
        left_stick_x = -1.0
    elif keys[pygame.K_RIGHT] and not keys[pygame.K_LEFT]:
        left_stick_x = 1.0

    left_stick_y = 0.0
    if keys[pygame.K_UP] and not keys[pygame.K_DOWN]:
        left_stick_y = -1.0
    elif keys[pygame.K_DOWN] and not keys[pygame.K_UP]:
        left_stick_y = 1.0

    return ViewerInput(
        quit_requested=quit_requested,
        toggle_pause=toggle_pause,
        step_once=step_once,
        save_state=save_state,
        force_reset=force_reset,
        toggle_deterministic_policy=toggle_deterministic_policy,
        toggle_manual_control=toggle_manual_control,
        control_fps_delta=control_fps_delta,
        panel_tab_delta=panel_tab_delta,
        panel_tab_index=panel_tab_index,
        control_state=ControllerState(
            joypad_mask=manual_mask,
            left_stick_x=left_stick_x,
            left_stick_y=left_stick_y,
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


def _clicked_panel_tab_index(
    position: object,
    rects: tuple[MouseRect | None, ...],
) -> int | None:
    for index, rect in enumerate(rects):
        if _point_in_rect(position, rect):
            return index
    return None
