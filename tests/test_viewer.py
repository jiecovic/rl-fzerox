# tests/test_viewer.py
import os

import numpy as np
import pygame

from rl_fzerox._native import JOYPAD_A, JOYPAD_START, JOYPAD_UP
from rl_fzerox.core.emulator.control import ControllerState
from rl_fzerox.core.emulator.video import display_size
from rl_fzerox.core.game import FZeroXTelemetry, PlayerTelemetry
from rl_fzerox.ui.viewer import (
    _build_panel_columns,
    _create_fonts,
    _panel_content_height,
    _pressed_button_labels,
    _window_size,
)


def test_target_display_size_applies_aspect_correction() -> None:
    frame_shape = np.zeros((240, 640, 3), dtype=np.uint8).shape

    corrected_display_size = display_size(frame_shape, 4.0 / 3.0)

    assert corrected_display_size == (640, 480)


def test_target_display_size_falls_back_to_raw_frame_size() -> None:
    frame_shape = np.zeros((240, 640, 3), dtype=np.uint8).shape

    corrected_display_size = display_size(frame_shape, 0.0)

    assert corrected_display_size == (640, 240)


def test_window_size_adds_sidebar_width() -> None:
    assert _window_size((640, 480)) == (1096, 480)


def test_pressed_button_labels_are_human_readable() -> None:
    assert _pressed_button_labels(0) == "none"
    assert _pressed_button_labels(
        (1 << JOYPAD_UP) | (1 << JOYPAD_A) | (1 << JOYPAD_START)
    ) == "Up A Start"


def test_side_panel_fits_default_480p_watch_window() -> None:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        fonts = _create_fonts(pygame)
        columns = _build_panel_columns(
            episode=0,
            info={
                "frame_index": 1592,
                "native_fps": 60.0,
                "display_aspect_ratio": 4.0 / 3.0,
            },
            reset_info={
                "reset_mode": "boot",
                "baseline_kind": "startup",
                "boot_state": "gp_race",
            },
            episode_reward=0.0,
            paused=False,
            control_state=ControllerState(),
            game_display_size=(640, 480),
            telemetry=_sample_telemetry(),
        )

        assert _panel_content_height(fonts, columns) <= 480
    finally:
        pygame.quit()


def _sample_telemetry() -> FZeroXTelemetry:
    return FZeroXTelemetry(
        system_ram_size=0x00800000,
        game_frame_count=1290,
        game_mode_raw=1,
        game_mode_name="gp_race",
        course_index=0,
        in_race_mode=True,
        player=PlayerTelemetry(
            state_flags=1 << 30,
            state_labels=("active",),
            speed_raw=0.0,
            speed_kph=0.0,
            energy=178.0,
            max_energy=178.0,
            boost_timer=0,
            race_distance=-3040.8,
            laps_completed_distance=-79028.0,
            lap_distance=75987.2,
            race_distance_position=-3040.8,
            race_time_ms=116,
            lap=1,
            laps_completed=0,
            position=30,
            character=0,
            machine_index=0,
        ),
    )
