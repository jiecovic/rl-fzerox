# src/rl_fzerox/ui/watch/input.py
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol

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
from rl_fzerox.ui.watch.view.screen.types import MouseRect, PygameModule, RecordCourseHitbox


class _PressedKeyState(Protocol):
    def __getitem__(self, key: int, /) -> bool: ...


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
    toggle_cnn_normalization: bool = False
    control_fps_delta: int = 0
    reset_control_fps: bool = False
    panel_tab_delta: int = 0
    panel_tab_index: int | None = None
    record_tab_index: int | None = None
    toggle_record_course_lock_id: str | None = None
    control_state: ControllerState = ControllerState()


@dataclass
class SpeedKeyRepeat:
    """Repeat timer for held watch speed-adjustment keys."""

    initial_delay_seconds: float = 0.25
    interval_seconds: float = 0.08
    _held_direction: int = 0
    _next_repeat_at: float = 0.0

    def delta(self, direction: int, *, now_seconds: float) -> int:
        """Return repeated speed steps for the currently held direction."""

        direction = _sign(direction)
        if direction == 0:
            self._held_direction = 0
            self._next_repeat_at = 0.0
            return 0
        if direction != self._held_direction:
            self._held_direction = direction
            self._next_repeat_at = now_seconds + max(0.0, self.initial_delay_seconds)
            return 0
        if now_seconds < self._next_repeat_at:
            return 0

        interval = max(0.001, self.interval_seconds)
        count = 1 + int((now_seconds - self._next_repeat_at) // interval)
        self._next_repeat_at += count * interval
        return direction * count


def _poll_viewer_input(
    pygame: PygameModule,
    *,
    deterministic_toggle_rect: MouseRect | None = None,
    panel_tab_rects: tuple[MouseRect | None, ...] = (),
    record_tab_rects: tuple[MouseRect | None, ...] = (),
    record_course_hitboxes: tuple[RecordCourseHitbox, ...] = (),
    speed_repeat: SpeedKeyRepeat | None = None,
    now_seconds: float | None = None,
) -> ViewerInput:
    quit_requested = False
    toggle_pause = False
    step_once = False
    save_state = False
    force_reset = False
    toggle_deterministic_policy = False
    toggle_manual_control = False
    toggle_cnn_normalization = False
    control_fps_delta = 0
    reset_control_fps = False
    panel_tab_delta = 0
    panel_tab_index = None
    record_tab_index = None
    toggle_record_course_lock_id = None

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
                else:
                    selected_record_tab = _clicked_panel_tab_index(event.pos, record_tab_rects)
                    if selected_record_tab is not None:
                        record_tab_index = selected_record_tab
                    else:
                        toggle_record_course_lock_id = _clicked_record_course_id(
                            event.pos,
                            record_course_hitboxes,
                        )
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
            elif event.key == pygame.K_c:
                toggle_cnn_normalization = True
            elif event.key in (pygame.K_0, pygame.K_KP0):
                reset_control_fps = True
            elif event.key in (pygame.K_PLUS, pygame.K_KP_PLUS):
                control_fps_delta += 1
            elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                control_fps_delta -= 1

    keys: _PressedKeyState = pygame.key.get_pressed()
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

    if speed_repeat is not None:
        control_fps_delta += speed_repeat.delta(
            _held_speed_direction(pygame, keys),
            now_seconds=time.perf_counter() if now_seconds is None else now_seconds,
        )

    return ViewerInput(
        quit_requested=quit_requested,
        toggle_pause=toggle_pause,
        step_once=step_once,
        save_state=save_state,
        force_reset=force_reset,
        toggle_deterministic_policy=toggle_deterministic_policy,
        toggle_manual_control=toggle_manual_control,
        toggle_cnn_normalization=toggle_cnn_normalization,
        control_fps_delta=control_fps_delta,
        reset_control_fps=reset_control_fps,
        panel_tab_delta=panel_tab_delta,
        panel_tab_index=panel_tab_index,
        record_tab_index=record_tab_index,
        toggle_record_course_lock_id=toggle_record_course_lock_id,
        control_state=ControllerState(
            joypad_mask=manual_mask,
            left_stick_x=left_stick_x,
            left_stick_y=left_stick_y,
        ),
    )


def _held_speed_direction(pygame: PygameModule, keys: _PressedKeyState) -> int:
    plus_held = keys[pygame.K_PLUS] or keys[pygame.K_KP_PLUS]
    minus_held = keys[pygame.K_MINUS] or keys[pygame.K_KP_MINUS]
    if plus_held and not minus_held:
        return 1
    if minus_held and not plus_held:
        return -1
    return 0


def _sign(value: int) -> int:
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


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


def _clicked_record_course_id(
    position: object,
    hitboxes: tuple[RecordCourseHitbox, ...],
) -> str | None:
    for hitbox in hitboxes:
        if _point_in_rect(position, hitbox.rect):
            return hitbox.course_id
    return None


def mouse_over_clickable(
    position: object,
    *,
    deterministic_toggle_rect: MouseRect | None = None,
    panel_tab_rects: tuple[MouseRect | None, ...] = (),
    record_tab_rects: tuple[MouseRect | None, ...] = (),
    record_course_hitboxes: tuple[RecordCourseHitbox, ...] = (),
) -> bool:
    """Return whether the mouse position is over a clickable watch UI target."""

    return (
        _point_in_rect(position, deterministic_toggle_rect)
        or _clicked_panel_tab_index(position, panel_tab_rects) is not None
        or _clicked_panel_tab_index(position, record_tab_rects) is not None
        or _clicked_record_course_id(position, record_course_hitboxes) is not None
    )
