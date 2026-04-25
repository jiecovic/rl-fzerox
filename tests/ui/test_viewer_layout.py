# tests/ui/test_viewer_layout.py
import os

import numpy as np
import pygame

from fzerox_emulator import ControllerState, display_size
from rl_fzerox.core.envs.observations import state_feature_names
from rl_fzerox.ui.watch.app import _next_panel_tab_index
from rl_fzerox.ui.watch.input import ViewerInput, _point_in_rect
from rl_fzerox.ui.watch.view.panels.model import (
    _build_panel_columns,
    _panel_content_height,
    _window_size,
)
from rl_fzerox.ui.watch.view.panels.section_renderer import _draw_labeled_value_line
from rl_fzerox.ui.watch.view.panels.tab_bar import _draw_panel_tabs, _panel_tab_hint
from rl_fzerox.ui.watch.view.screen.frame import _create_fonts
from rl_fzerox.ui.watch.view.screen.layout import LAYOUT
from rl_fzerox.ui.watch.view.screen.theme import PALETTE
from rl_fzerox.ui.watch.view.screen.types import PanelLine
from tests.ui.viewer_support import FakeScreen, fake_viewer_fonts
from tests.ui.viewer_support import sample_telemetry as _sample_telemetry


def test_panel_value_rows_keep_stable_height_when_glyph_height_changes() -> None:
    fonts = fake_viewer_fonts()
    screen = FakeScreen()
    common_kwargs = {
        "pygame": None,
        "screen": screen,
        "fonts": fonts,
        "x": 0,
        "y": 10,
        "width": 160,
    }

    no_y = _draw_labeled_value_line(
        **common_kwargs,
        line=PanelLine("Sliding", "no", PALETTE.text_muted),
    )
    yes_y = _draw_labeled_value_line(
        **common_kwargs,
        line=PanelLine("Sliding", "yes", PALETTE.text_warning),
    )

    assert no_y == yes_y


def test_point_in_rect_matches_clickable_region_bounds() -> None:
    rect = (10, 20, 30, 40)

    assert _point_in_rect((10, 20), rect) is True
    assert _point_in_rect((39, 59), rect) is True
    assert _point_in_rect((40, 59), rect) is False
    assert _point_in_rect((39, 60), rect) is False
    assert _point_in_rect((10, 20), None) is False


def test_target_display_size_applies_aspect_correction() -> None:
    frame_shape = np.zeros((240, 640, 3), dtype=np.uint8).shape

    corrected_display_size = display_size(frame_shape, 4.0 / 3.0)

    assert corrected_display_size == (640, 480)


def test_target_display_size_falls_back_to_raw_frame_size() -> None:
    frame_shape = np.zeros((240, 640, 3), dtype=np.uint8).shape

    corrected_display_size = display_size(frame_shape, 0.0)

    assert corrected_display_size == (640, 240)


def test_next_panel_tab_index_cycles_tabs() -> None:
    assert _next_panel_tab_index(0, ViewerInput(panel_tab_delta=1)) == 1
    assert _next_panel_tab_index(5, ViewerInput(panel_tab_delta=1)) == 0


def test_next_panel_tab_index_honors_direct_selection() -> None:
    assert _next_panel_tab_index(0, ViewerInput(panel_tab_index=2)) == 2


def test_panel_tab_hint_shows_active_tab_position() -> None:
    assert _panel_tab_hint(0) == "Tab 1/6"
    assert _panel_tab_hint(2) == "Tab 3/6"
    assert _panel_tab_hint(3) == "Tab 4/6"
    assert _panel_tab_hint(4) == "Tab 5/6"
    assert _panel_tab_hint(5) == "Tab 6/6"
    assert _panel_tab_hint(6) == "Tab 1/6"


def test_panel_tabs_fit_side_panel_content_width() -> None:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    pygame.init()
    try:
        fonts = _create_fonts(pygame)
        width = LAYOUT.panel_width - (2 * LAYOUT.panel_padding)
        screen = pygame.Surface((LAYOUT.panel_width, 80))

        _, tab_rects = _draw_panel_tabs(
            pygame=pygame,
            screen=screen,
            fonts=fonts,
            x=0,
            y=0,
            width=width,
            selected_index=0,
        )

        assert all(rect is not None for rect in tab_rects)
        assert max(rect[0] + rect[2] for rect in tab_rects if rect is not None) <= width
    finally:
        pygame.quit()


def test_window_size_adds_sidebar_width() -> None:
    assert _window_size((592, 444), (84, 116, 12)) == (1004, 980)
    assert _window_size((592, 444), (84, 116, 12), panel_tab_index=0) == (1004, 980)
    assert _window_size((592, 444), (84, 116, 12), panel_tab_index=1) == (1004, 980)
    assert _window_size((592, 444), (84, 116, 12), panel_tab_index=2) == (1004, 980)
    assert _window_size((592, 444), (84, 116, 12), panel_tab_index=3) == (1004, 980)
    assert _window_size((592, 444), (84, 116, 12), panel_tab_index=4) == (1004, 980)
    assert _window_size((592, 444), (84, 116, 12), panel_tab_index=5) == (1004, 980)


def test_side_panel_fits_default_watch_window_height() -> None:
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
                "observation_stack": 4,
            },
            reset_info={
                "reset_mode": "boot",
                "baseline_kind": "startup",
                "boot_state": "gp_race",
            },
            episode_reward=0.0,
            paused=False,
            control_state=ControllerState(),
            policy_curriculum_stage=None,
            policy_action=None,
            policy_reload_age_seconds=None,
            policy_reload_error=None,
            action_repeat=3,
            stuck_min_speed_kph=50.0,
            game_display_size=(592, 444),
            observation_shape=(84, 116, 12),
            telemetry=_sample_telemetry(),
        )

        assert (
            _panel_content_height(
                fonts,
                columns,
                observation_shape=(84, 116, 12),
            )
            <= _window_size((592, 444), (84, 116, 12), panel_tab_index=1)[1]
        )
    finally:
        pygame.quit()


def test_side_panel_fits_steer_history_observation_state_vector() -> None:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        fonts = _create_fonts(pygame)
        feature_names = state_feature_names("steer_history")
        observation_shape = (84, 116, 3)
        columns = _build_panel_columns(
            episode=0,
            info={"frame_index": 0, "native_fps": 60.0, "display_aspect_ratio": 4.0 / 3.0},
            reset_info={},
            episode_reward=0.0,
            paused=False,
            control_state=ControllerState(),
            policy_curriculum_stage=None,
            policy_action=np.array([1, 1, 0, 0, 0], dtype=np.int64),
            policy_reload_age_seconds=0.0,
            policy_reload_error=None,
            action_repeat=1,
            stuck_min_speed_kph=60.0,
            game_display_size=(592, 444),
            observation_shape=observation_shape,
            observation_state=np.zeros((len(feature_names),), dtype=np.float32),
            observation_state_feature_names=feature_names,
            telemetry=_sample_telemetry(),
        )

        assert (
            _panel_content_height(
                fonts,
                columns,
                observation_shape=observation_shape,
            )
            <= _window_size((592, 444), observation_shape, panel_tab_index=2)[1]
        )
    finally:
        pygame.quit()
