# tests/ui/test_viewer_layout.py
import os
from pathlib import Path

import numpy as np
import pygame

from fzerox_emulator import RaceControlState, display_size
from rl_fzerox.core.runtime_spec.schema import EmulatorConfig, WatchAppConfig, WatchConfig
from rl_fzerox.ui.watch.app import (
    _initial_policy_observation_layout_shape,
    _next_panel_tab_index,
    _next_policy_observation_layout_shape,
    _policy_observation_layout_info,
)
from rl_fzerox.ui.watch.input import ViewerInput, _point_in_rect
from rl_fzerox.ui.watch.view.components.game_view import _draw_glass_game_view
from rl_fzerox.ui.watch.view.components.observation_strip import (
    _draw_observation_preview_in_rect,
)
from rl_fzerox.ui.watch.view.panels.core.model import (
    _build_panel_columns,
    _panel_content_height,
    _window_size,
)
from rl_fzerox.ui.watch.view.panels.core.tabs import CAREER_PANEL_TABS
from rl_fzerox.ui.watch.view.panels.rendering.section_renderer import (
    _draw_labeled_value_line,
)
from rl_fzerox.ui.watch.view.panels.rendering.tab_bar import (
    _draw_panel_tabs,
    _panel_tab_hint,
)
from rl_fzerox.ui.watch.view.screen.frame import _create_fonts, _watch_window_size
from rl_fzerox.ui.watch.view.screen.layout import LAYOUT
from rl_fzerox.ui.watch.view.screen.theme import PALETTE
from rl_fzerox.ui.watch.view.screen.types import PanelLine
from tests.ui.viewer_support import FakeScreen, fake_viewer_fonts
from tests.ui.viewer_support import sample_telemetry as _sample_telemetry


class _FakePygame:
    @staticmethod
    def Rect(x: int, y: int, width: int, height: int) -> tuple[int, int, int, int]:
        return (x, y, width, height)

    class draw:
        @staticmethod
        def circle(*args: object, **kwargs: object) -> None:
            return None

        @staticmethod
        def line(*args: object, **kwargs: object) -> None:
            return None

        @staticmethod
        def polygon(*args: object, **kwargs: object) -> None:
            return None

        @staticmethod
        def rect(*args: object, **kwargs: object) -> None:
            return None


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


def test_panel_value_rows_with_status_icons_still_draw_values() -> None:
    fonts = fake_viewer_fonts()
    screen = FakeScreen()

    _draw_labeled_value_line(
        pygame=_FakePygame,
        screen=screen,
        fonts=fonts,
        x=0,
        y=10,
        width=220,
        line=PanelLine(
            "speed_kph_norm",
            "0.500",
            PALETTE.text_primary,
            status_icon="toggle_on",
            click_state_feature_name="speed_kph_norm",
        ),
    )

    assert "speed_kph_norm" in [text for text, _ in screen.blits]
    assert "0.500" in [text for text, _ in screen.blits]


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
    assert _next_panel_tab_index(8, ViewerInput(panel_tab_delta=1)) == 0


def test_next_panel_tab_index_honors_direct_selection() -> None:
    assert _next_panel_tab_index(0, ViewerInput(panel_tab_index=2)) == 2


def test_menu_snapshot_keeps_previous_policy_observation_layout_shape() -> None:
    class _Snapshot:
        def __init__(self, shape: tuple[int, ...] | None) -> None:
            self.policy_observation_shape = shape

    policy_shape = (72, 96, 6)
    menu_snapshot = _Snapshot(None)

    assert _initial_policy_observation_layout_shape(menu_snapshot) == (72, 96, 3)
    assert _next_policy_observation_layout_shape(policy_shape, menu_snapshot) == policy_shape
    assert _next_policy_observation_layout_shape(
        policy_shape,
        _Snapshot((84, 84, 12)),
    ) == (84, 84, 12)


def test_policy_observation_layout_shape_hint_stabilizes_menu_layout(
    tmp_path: Path,
) -> None:
    class _Snapshot:
        def __init__(self, shape: tuple[int, ...] | None) -> None:
            self.policy_observation_shape = shape

    core_path = tmp_path / "core.so"
    rom_path = tmp_path / "rom.n64"
    core_path.touch()
    rom_path.touch()
    config = WatchAppConfig(
        emulator=EmulatorConfig(core_path=core_path, rom_path=rom_path),
        watch=WatchConfig(policy_observation_layout_shape_hint=(100, 240, 12)),
    )

    assert _initial_policy_observation_layout_shape(_Snapshot(None), config=config) == (
        100,
        240,
        12,
    )
    assert _next_policy_observation_layout_shape(
        (100, 240, 12),
        _Snapshot((72, 96, 6)),
        config=config,
    ) == (100, 240, 12)

    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    pygame.init()
    try:
        fonts = _create_fonts(pygame)
        layout_info = _policy_observation_layout_info(config)
        assert layout_info is not None
        stable_size = _watch_window_size(
            (592, 444),
            (100, 240, 12),
            fonts=fonts,
            info=layout_info,
        )
        live_policy_info = {"observation_stack": 2, "observation_stack_mode": "rgb"}

        assert (
            _watch_window_size(
                (592, 444),
                (100, 240, 12),
                fonts=fonts,
                info=layout_info,
            )
            == stable_size
        )
        assert (
            _watch_window_size(
                (592, 444),
                (100, 240, 12),
                fonts=fonts,
                info=live_policy_info,
            )
            == stable_size
        )
    finally:
        pygame.quit()


def test_panel_tab_hint_shows_active_tab_position() -> None:
    assert _panel_tab_hint(0) == "Tab 1/9"
    assert _panel_tab_hint(2) == "Tab 3/9"
    assert _panel_tab_hint(3) == "Tab 4/9"
    assert _panel_tab_hint(4) == "Tab 5/9"
    assert _panel_tab_hint(5) == "Tab 6/9"
    assert _panel_tab_hint(6) == "Tab 7/9"
    assert _panel_tab_hint(7) == "Tab 8/9"
    assert _panel_tab_hint(8) == "Tab 9/9"
    assert _panel_tab_hint(9) == "Tab 1/9"


def test_career_panel_tabs_include_records_and_career() -> None:
    assert CAREER_PANEL_TABS.labels == (
        "Run",
        "Live",
        "Obs",
        "Details",
        "State",
        "Aux",
        "CNN",
        "Records",
        "Career",
        "Train",
    )
    assert CAREER_PANEL_TABS.records_index == 7
    assert CAREER_PANEL_TABS.career_index == 8
    assert _panel_tab_hint(8, panel_tabs=CAREER_PANEL_TABS) == "Tab 9/10"


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
    assert _window_size((592, 444), (84, 116, 12)) == (1204, 980)
    assert _window_size((592, 444), (84, 116, 12), panel_tab_index=0) == (1204, 980)
    assert _window_size((592, 444), (84, 116, 12), panel_tab_index=1) == (1204, 980)
    assert _window_size((592, 444), (84, 116, 12), panel_tab_index=2) == (1204, 980)
    assert _window_size((592, 444), (84, 116, 12), panel_tab_index=3) == (1204, 980)
    assert _window_size((592, 444), (84, 116, 12), panel_tab_index=4) == (1204, 980)
    assert _window_size((592, 444), (84, 116, 12), panel_tab_index=5) == (1204, 980)
    assert _window_size((592, 444), (84, 116, 12), panel_tab_index=6) == (1204, 980)
    assert _window_size((592, 444), (84, 116, 12), panel_tab_index=7) == (1204, 980)
    assert _window_size((592, 444), (84, 116, 12), panel_tab_index=8) == (1204, 980)


def test_watch_window_size_ignores_policy_observation_preview_size() -> None:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        fonts = _create_fonts(pygame)
        default_size = _watch_window_size(
            (592, 444),
            (72, 96, 3),
            fonts=fonts,
            info={},
        )
        wide_policy_size = _watch_window_size(
            (592, 444),
            (180, 240, 6),
            fonts=fonts,
            info={"observation_stack": 2, "observation_stack_mode": "rgb"},
        )

        assert wide_policy_size == default_size
        assert default_size[0] == 592 + LAYOUT.preview_gap + LAYOUT.panel_width
    finally:
        pygame.quit()


def test_watch_window_size_keeps_layout_when_policy_preview_is_hidden() -> None:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        fonts = _create_fonts(pygame)
        with_preview = _watch_window_size(
            (592, 444),
            (180, 240, 6),
            fonts=fonts,
            info={
                "observation_stack": 2,
                "observation_stack_mode": "rgb",
            },
        )
        without_preview = _watch_window_size(
            (592, 444),
            (180, 240, 6),
            fonts=fonts,
            info={
                "observation_stack": 2,
                "observation_stack_mode": "rgb",
            },
        )

        assert without_preview == with_preview
    finally:
        pygame.quit()


def test_observation_preview_draws_blank_panel_without_policy_image() -> None:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        fonts = _create_fonts(pygame)
        screen = pygame.Surface((420, 260))
        screen.fill(PALETTE.app_background)
        before = pygame.surfarray.array3d(screen).copy()

        _draw_observation_preview_in_rect(
            pygame=pygame,
            screen=screen,
            fonts=fonts,
            surface=None,
            x=12,
            y=12,
            width=396,
            height=236,
            observation_shape=(84, 84, 12),
            layout_shape=(84, 84, 12),
            info={
                "observation_stack": 4,
                "observation_stack_mode": "rgb",
            },
        )

        after = pygame.surfarray.array3d(screen)
        assert np.any(after != before)
    finally:
        pygame.quit()


def test_game_view_draws_recording_dot_when_active() -> None:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        fonts = _create_fonts(pygame)
        screen = pygame.Surface((160, 120))
        game_surface = pygame.Surface((132, 96))
        game_surface.fill((20, 20, 20))

        _draw_glass_game_view(
            pygame=pygame,
            screen=screen,
            fonts=fonts,
            surface=game_surface,
            outer_size=(160, 120),
            recording_active=True,
        )

        red, green, blue, _ = screen.get_at((33, 31))
        assert red > 200
        assert green < 100
        assert blue < 100
    finally:
        pygame.quit()


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
            control_state=RaceControlState(),
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
        feature_names = (
            "vehicle_state.speed_norm",
            "vehicle_state.energy_frac",
            "vehicle_state.reverse_active",
            "vehicle_state.airborne",
            "vehicle_state.can_boost",
            "vehicle_state.boost_active",
            "control_history.prev_steer_1",
            "control_history.prev_lean_1",
        )
        observation_shape = (84, 116, 3)
        columns = _build_panel_columns(
            episode=0,
            info={"frame_index": 0, "native_fps": 60.0, "display_aspect_ratio": 4.0 / 3.0},
            reset_info={},
            episode_reward=0.0,
            paused=False,
            control_state=RaceControlState(),
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
