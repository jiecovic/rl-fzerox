# tests/ui/test_watch_input.py
from __future__ import annotations

from types import SimpleNamespace

from rl_fzerox.ui.watch.input import _poll_viewer_input


class _InactiveKeys:
    def __getitem__(self, _key: int) -> bool:
        return False


class _FakePygame:
    QUIT = 1
    KEYDOWN = 2
    K_p = 10
    K_n = 11
    K_k = 12
    K_PLUS = 13
    K_KP_PLUS = 14
    K_MINUS = 15
    K_KP_MINUS = 16
    K_EQUALS = 17
    K_r = 26
    K_UP = 18
    K_DOWN = 19
    K_LEFT = 20
    K_RIGHT = 21
    K_x = 22
    K_z = 23
    K_RETURN = 24
    K_BACKSPACE = 25

    def __init__(self, key_events: tuple[int, ...]) -> None:
        self.event = SimpleNamespace(
            get=lambda: [SimpleNamespace(type=self.KEYDOWN, key=key) for key in key_events]
        )
        self.key = SimpleNamespace(get_pressed=lambda: _InactiveKeys())


def test_poll_viewer_input_adjusts_control_fps_with_plus_keys() -> None:
    viewer_input = _poll_viewer_input(_FakePygame((_FakePygame.K_PLUS, _FakePygame.K_KP_PLUS)))

    assert viewer_input.control_fps_delta == 2


def test_poll_viewer_input_adjusts_control_fps_with_minus_keys() -> None:
    viewer_input = _poll_viewer_input(_FakePygame((_FakePygame.K_MINUS, _FakePygame.K_KP_MINUS)))

    assert viewer_input.control_fps_delta == -2


def test_poll_viewer_input_does_not_treat_equals_as_plus() -> None:
    viewer_input = _poll_viewer_input(_FakePygame((_FakePygame.K_EQUALS,)))

    assert viewer_input.control_fps_delta == 0


def test_poll_viewer_input_maps_r_to_force_reset() -> None:
    viewer_input = _poll_viewer_input(_FakePygame((_FakePygame.K_r,)))

    assert viewer_input.force_reset is True
