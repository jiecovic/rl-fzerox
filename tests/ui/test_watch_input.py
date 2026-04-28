# tests/ui/test_watch_input.py
from __future__ import annotations

from types import SimpleNamespace

from rl_fzerox.core.envs.actions import (
    ACCELERATE_MASK,
    AIR_BRAKE_MASK,
    BOOST_MASK,
    LEAN_LEFT_MASK,
    LEAN_RIGHT_MASK,
)
from rl_fzerox.ui.watch.input import SpeedKeyRepeat, _poll_viewer_input, mouse_over_clickable
from rl_fzerox.ui.watch.view.screen.types import RecordCourseHitbox


class _PressedKeys:
    def __init__(self, pressed: tuple[int, ...] = ()) -> None:
        self._pressed = frozenset(pressed)

    def __getitem__(self, _key: int) -> bool:
        return _key in self._pressed


class _FakePygame:
    QUIT = 1
    KEYDOWN = 2
    MOUSEBUTTONDOWN = 3
    K_ESCAPE = 9
    K_TAB = 28
    K_p = 10
    K_n = 11
    K_k = 12
    K_PLUS = 13
    K_KP_PLUS = 14
    K_MINUS = 15
    K_KP_MINUS = 16
    K_EQUALS = 17
    K_r = 26
    K_d = 27
    K_m = 35
    K_c = 39
    K_a = 36
    K_s = 37
    K_SPACE = 38
    K_UP = 18
    K_DOWN = 19
    K_LEFT = 20
    K_RIGHT = 21
    K_x = 22
    K_z = 23
    K_RETURN = 24
    K_BACKSPACE = 25
    K_1 = 29
    K_2 = 30
    K_3 = 31
    K_4 = 32
    K_5 = 33
    K_6 = 34

    def __init__(
        self,
        key_events: tuple[int, ...],
        *,
        pressed_keys: tuple[int, ...] = (),
        mouse_click: tuple[int, int] | None = None,
    ) -> None:
        events = [SimpleNamespace(type=self.KEYDOWN, key=key) for key in key_events]
        if mouse_click is not None:
            events.append(
                SimpleNamespace(
                    type=self.MOUSEBUTTONDOWN,
                    button=1,
                    pos=mouse_click,
                )
            )
        self.event = SimpleNamespace(get=lambda: events)
        self.key = SimpleNamespace(get_pressed=lambda: _PressedKeys(pressed_keys))


def test_poll_viewer_input_adjusts_control_fps_with_plus_keys() -> None:
    viewer_input = _poll_viewer_input(_FakePygame((_FakePygame.K_PLUS, _FakePygame.K_KP_PLUS)))

    assert viewer_input.control_fps_delta == 2


def test_poll_viewer_input_adjusts_control_fps_with_minus_keys() -> None:
    viewer_input = _poll_viewer_input(_FakePygame((_FakePygame.K_MINUS, _FakePygame.K_KP_MINUS)))

    assert viewer_input.control_fps_delta == -2


def test_poll_viewer_input_does_not_treat_equals_as_plus() -> None:
    viewer_input = _poll_viewer_input(_FakePygame((_FakePygame.K_EQUALS,)))

    assert viewer_input.control_fps_delta == 0


def test_poll_viewer_input_repeats_control_fps_delta_while_plus_is_held() -> None:
    repeat = SpeedKeyRepeat(initial_delay_seconds=0.2, interval_seconds=0.1)

    first = _poll_viewer_input(
        _FakePygame((), pressed_keys=(_FakePygame.K_PLUS,)),
        speed_repeat=repeat,
        now_seconds=1.0,
    )
    before_delay = _poll_viewer_input(
        _FakePygame((), pressed_keys=(_FakePygame.K_PLUS,)),
        speed_repeat=repeat,
        now_seconds=1.19,
    )
    after_delay = _poll_viewer_input(
        _FakePygame((), pressed_keys=(_FakePygame.K_PLUS,)),
        speed_repeat=repeat,
        now_seconds=1.2,
    )
    later = _poll_viewer_input(
        _FakePygame((), pressed_keys=(_FakePygame.K_PLUS,)),
        speed_repeat=repeat,
        now_seconds=1.45,
    )

    assert first.control_fps_delta == 0
    assert before_delay.control_fps_delta == 0
    assert after_delay.control_fps_delta == 1
    assert later.control_fps_delta == 2


def test_poll_viewer_input_repeats_control_fps_delta_while_minus_is_held() -> None:
    repeat = SpeedKeyRepeat(initial_delay_seconds=0.0, interval_seconds=0.1)
    _poll_viewer_input(
        _FakePygame((), pressed_keys=(_FakePygame.K_MINUS,)),
        speed_repeat=repeat,
        now_seconds=1.0,
    )

    viewer_input = _poll_viewer_input(
        _FakePygame((), pressed_keys=(_FakePygame.K_MINUS,)),
        speed_repeat=repeat,
        now_seconds=1.0,
    )

    assert viewer_input.control_fps_delta == -1


def test_poll_viewer_input_does_not_double_count_initial_held_speed_key() -> None:
    repeat = SpeedKeyRepeat(initial_delay_seconds=0.2, interval_seconds=0.1)

    viewer_input = _poll_viewer_input(
        _FakePygame((_FakePygame.K_PLUS,), pressed_keys=(_FakePygame.K_PLUS,)),
        speed_repeat=repeat,
        now_seconds=1.0,
    )

    assert viewer_input.control_fps_delta == 1


def test_poll_viewer_input_maps_r_to_force_reset() -> None:
    viewer_input = _poll_viewer_input(_FakePygame((_FakePygame.K_r,)))

    assert viewer_input.force_reset is True


def test_poll_viewer_input_maps_d_to_policy_mode_toggle() -> None:
    viewer_input = _poll_viewer_input(_FakePygame((_FakePygame.K_d,)))

    assert viewer_input.toggle_deterministic_policy is True


def test_poll_viewer_input_maps_m_to_manual_control_toggle() -> None:
    viewer_input = _poll_viewer_input(_FakePygame((_FakePygame.K_m,)))

    assert viewer_input.toggle_manual_control is True


def test_poll_viewer_input_maps_c_to_cnn_normalization_toggle() -> None:
    viewer_input = _poll_viewer_input(_FakePygame((_FakePygame.K_c,)))

    assert viewer_input.toggle_cnn_normalization is True


def test_poll_viewer_input_maps_escape_to_quit() -> None:
    viewer_input = _poll_viewer_input(_FakePygame((_FakePygame.K_ESCAPE,)))

    assert viewer_input.quit_requested is True


def test_poll_viewer_input_cycles_panel_tabs_with_tab() -> None:
    viewer_input = _poll_viewer_input(_FakePygame((_FakePygame.K_TAB,)))

    assert viewer_input.panel_tab_delta == 1


def test_poll_viewer_input_selects_panel_tabs_with_number_keys() -> None:
    assert _poll_viewer_input(_FakePygame((_FakePygame.K_1,))).panel_tab_index == 0
    assert _poll_viewer_input(_FakePygame((_FakePygame.K_2,))).panel_tab_index == 1
    assert _poll_viewer_input(_FakePygame((_FakePygame.K_3,))).panel_tab_index == 2
    assert _poll_viewer_input(_FakePygame((_FakePygame.K_4,))).panel_tab_index == 3
    assert _poll_viewer_input(_FakePygame((_FakePygame.K_5,))).panel_tab_index == 4
    assert _poll_viewer_input(_FakePygame((_FakePygame.K_6,))).panel_tab_index == 5


def test_poll_viewer_input_selects_panel_tab_with_mouse_click() -> None:
    viewer_input = _poll_viewer_input(
        _FakePygame((), mouse_click=(25, 12)),
        panel_tab_rects=((0, 0, 40, 20), (50, 0, 40, 20), None),
    )

    assert viewer_input.panel_tab_index == 0


def test_poll_viewer_input_selects_record_tab_with_mouse_click() -> None:
    viewer_input = _poll_viewer_input(
        _FakePygame((), mouse_click=(75, 42)),
        panel_tab_rects=((0, 0, 40, 20),),
        record_tab_rects=((0, 30, 40, 20), (50, 30, 40, 20)),
    )

    assert viewer_input.record_tab_index == 1


def test_poll_viewer_input_selects_record_course_with_mouse_click() -> None:
    viewer_input = _poll_viewer_input(
        _FakePygame((), mouse_click=(75, 72)),
        panel_tab_rects=((0, 0, 40, 20),),
        record_tab_rects=((0, 30, 40, 20),),
        record_course_hitboxes=(
            RecordCourseHitbox(rect=(50, 60, 120, 20), course_id="mute_city"),
        ),
    )

    assert viewer_input.toggle_record_course_lock_id == "mute_city"


def test_mouse_over_clickable_matches_watch_hitboxes() -> None:
    record_hitbox = RecordCourseHitbox(rect=(50, 60, 120, 20), course_id="mute_city")

    assert mouse_over_clickable(
        (25, 12),
        deterministic_toggle_rect=(0, 0, 40, 20),
    )
    assert mouse_over_clickable(
        (75, 42),
        record_tab_rects=((50, 30, 40, 20),),
    )
    assert mouse_over_clickable((75, 72), record_course_hitboxes=(record_hitbox,))
    assert not mouse_over_clickable(
        (500, 500),
        deterministic_toggle_rect=(0, 0, 40, 20),
        panel_tab_rects=((0, 30, 40, 20),),
        record_tab_rects=((50, 30, 40, 20),),
        record_course_hitboxes=(record_hitbox,),
    )


def test_poll_viewer_input_maps_manual_keys_to_n64_controls() -> None:
    viewer_input = _poll_viewer_input(
        _FakePygame(
            (),
            pressed_keys=(
                _FakePygame.K_LEFT,
                _FakePygame.K_UP,
                _FakePygame.K_z,
                _FakePygame.K_x,
                _FakePygame.K_SPACE,
                _FakePygame.K_a,
                _FakePygame.K_s,
            ),
        )
    )

    state = viewer_input.control_state
    assert state.left_stick_x == -1.0
    assert state.left_stick_y == -1.0
    assert state.joypad_mask & ACCELERATE_MASK
    assert state.joypad_mask & AIR_BRAKE_MASK
    assert state.joypad_mask & BOOST_MASK
    assert state.joypad_mask & LEAN_LEFT_MASK
    assert state.joypad_mask & LEAN_RIGHT_MASK
