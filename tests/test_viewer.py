# tests/test_viewer.py
import os

import numpy as np
import pygame

from rl_fzerox._native import JOYPAD_A, JOYPAD_START, JOYPAD_UP
from rl_fzerox.core.emulator.video import display_size
from rl_fzerox.ui.viewer import (
    _build_panel_sections,
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
    assert _window_size((640, 480)) == (976, 480)


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
        sections = _build_panel_sections(
            episode=0,
            info={
                "frame_index": 1592,
                "native_fps": 60.0,
                "display_aspect_ratio": 4.0 / 3.0,
            },
            reset_info={
                "reset_mode": "boot",
                "baseline_kind": "startup",
                "boot_state": "race_grid",
            },
            episode_reward=0.0,
            paused=False,
            joypad_mask=0,
            game_display_size=(640, 480),
        )

        assert _panel_content_height(fonts, sections) <= 480
    finally:
        pygame.quit()
