# tests/ui/test_viewer.py
import os

import numpy as np
import pygame

from fzerox_emulator import (
    JOYPAD_A,
    JOYPAD_START,
    JOYPAD_UP,
    ControllerState,
    FZeroXTelemetry,
    display_size,
)
from rl_fzerox.core.envs.actions import BOOST_MASK, DRIFT_LEFT_MASK, THROTTLE_MASK
from rl_fzerox.ui.watch import (
    _build_panel_columns,
    _create_fonts,
    _format_policy_action,
    _format_reload_age,
    _format_reload_error,
    _panel_content_height,
    _persist_reload_error,
    _pressed_button_labels,
    _preview_frame,
    _window_size,
)
from tests.support.native_objects import make_telemetry


def test_target_display_size_applies_aspect_correction() -> None:
    frame_shape = np.zeros((240, 640, 3), dtype=np.uint8).shape

    corrected_display_size = display_size(frame_shape, 4.0 / 3.0)

    assert corrected_display_size == (640, 480)


def test_target_display_size_falls_back_to_raw_frame_size() -> None:
    frame_shape = np.zeros((240, 640, 3), dtype=np.uint8).shape

    corrected_display_size = display_size(frame_shape, 0.0)

    assert corrected_display_size == (640, 240)


def test_window_size_adds_sidebar_width() -> None:
    assert _window_size((592, 444), (78, 222, 12)) == (1060, 720)


def test_pressed_button_labels_are_human_readable() -> None:
    assert _pressed_button_labels(0) == "none"
    assert _pressed_button_labels(
        (1 << JOYPAD_UP) | (1 << JOYPAD_A) | (1 << JOYPAD_START)
    ) == "Up A Start"


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
            policy_label=None,
            policy_action=None,
            policy_reload_age_seconds=None,
            policy_reload_error=None,
            action_repeat=3,
            stuck_step_limit=240,
            game_display_size=(592, 444),
            observation_shape=(78, 222, 12),
            telemetry=_sample_telemetry(),
        )

        assert _panel_content_height(
            fonts,
            columns,
            observation_shape=(78, 222, 12),
        ) <= _window_size((592, 444), (78, 222, 12))[1]
    finally:
        pygame.quit()


def test_input_section_includes_visualized_control_state() -> None:
    columns = _build_panel_columns(
        episode=0,
        info={"frame_index": 0, "native_fps": 60.0},
        reset_info={},
        episode_reward=0.0,
        paused=False,
        control_state=ControllerState(
            joypad_mask=THROTTLE_MASK | BOOST_MASK | DRIFT_LEFT_MASK,
            left_stick_x=0.5,
        ),
        policy_label=None,
        policy_action=None,
        policy_reload_age_seconds=None,
        policy_reload_error=None,
        action_repeat=3,
        stuck_step_limit=240,
        game_display_size=(592, 444),
        observation_shape=(78, 222, 12),
        telemetry=_sample_telemetry(),
    )

    input_section = next(section for section in columns.left if section.title == "Input")
    assert input_section.control_viz is not None
    assert input_section.control_viz.drive_level == 1
    assert input_section.control_viz.steer_x == 0.5
    assert input_section.control_viz.boost_pressed
    assert input_section.control_viz.drift_direction == -1


def test_game_flags_are_rendered_in_fixed_rows() -> None:
    columns = _build_panel_columns(
        episode=0,
        info={"frame_index": 0, "native_fps": 60.0},
        reset_info={},
        episode_reward=0.0,
        paused=False,
        control_state=ControllerState(),
        policy_label=None,
        policy_action=None,
        policy_reload_age_seconds=None,
        policy_reload_error=None,
        action_repeat=3,
        stuck_step_limit=240,
        game_display_size=(592, 444),
        observation_shape=(78, 222, 12),
        telemetry=_sample_telemetry(
            state_labels=("active", "dash_pad_boost", "collision_recoil", "wrong_way"),
        ),
    )

    game_section = columns.right[0]
    assert game_section.flag_viz is not None
    assert len(game_section.flag_viz.rows) == 3
    active_labels = {
        token.label
        for row in game_section.flag_viz.rows
        for token in row
        if token.active
    }
    assert {"dash", "recoil"}.issubset(active_labels)


def test_preview_frame_uses_latest_rgb_slice_for_stacked_rgb_observations() -> None:
    first = np.zeros((2, 3, 3), dtype=np.uint8)
    second = np.full((2, 3, 3), 255, dtype=np.uint8)
    stacked = np.concatenate((first, second), axis=2)

    preview = _preview_frame(stacked)

    assert preview.shape == (2, 3, 3)
    assert np.array_equal(preview, second)


def test_format_policy_action_is_human_readable() -> None:
    assert _format_policy_action(None) == "manual"
    assert _format_policy_action(np.array([2, 0], dtype=np.int64)) == "[2,0]"
    assert _format_policy_action(np.array([4, 1], dtype=np.int64)) == "[4,1]"
    assert _format_policy_action(np.array([4, 1, 1, 2], dtype=np.int64)) == "[4,1,1,2]"


def test_display_section_includes_action_repeat() -> None:
    columns = _build_panel_columns(
        episode=0,
        info={"frame_index": 0, "native_fps": 60.0},
        reset_info={},
        episode_reward=0.0,
        paused=False,
        control_state=ControllerState(),
        policy_label="ppo_cnn_0013",
        policy_action=np.array([2, 1, 0], dtype=np.int64),
        policy_reload_age_seconds=5.0,
        policy_reload_error=None,
        action_repeat=2,
        stuck_step_limit=240,
        game_display_size=(592, 444),
        observation_shape=(78, 222, 12),
        telemetry=_sample_telemetry(),
    )

    display_section = next(section for section in columns.right if section.title == "Display")
    repeat_line = next(line for line in display_section.lines if line.label == "Frame skip")

    assert repeat_line.value == "2"


def test_session_section_includes_stuck_counter() -> None:
    columns = _build_panel_columns(
        episode=0,
        info={"frame_index": 0, "native_fps": 60.0, "stalled_steps": 17},
        reset_info={},
        episode_reward=0.0,
        paused=False,
        control_state=ControllerState(),
        policy_label="ppo_cnn_0017",
        policy_action=np.array([2, 1, 0], dtype=np.int64),
        policy_reload_age_seconds=5.0,
        policy_reload_error=None,
        action_repeat=1,
        stuck_step_limit=240,
        game_display_size=(592, 444),
        observation_shape=(78, 222, 12),
        telemetry=_sample_telemetry(),
    )

    session_section = next(section for section in columns.left if section.title == "Session")
    stuck_line = next(line for line in session_section.lines if line.label == "Stuck")

    assert stuck_line.value == "17 / 240"


def test_format_reload_age_is_human_readable() -> None:
    assert _format_reload_age(None) == "manual"
    assert _format_reload_age(12.7) == "12s ago"
    assert _format_reload_age(125.0) == "2m 05s"
    assert _format_reload_age(3665.0) == "1h 01m"


def test_format_reload_error_truncates_long_messages() -> None:
    assert _format_reload_error(None) == "-"
    assert _format_reload_error("bad checkpoint") == "bad checkpoint"
    assert _format_reload_error("this is a much longer checkpoint parse failure message") == (
        "this is a much longer checkpoint pa…"
    )


def test_persist_reload_error_writes_full_message_once(tmp_path) -> None:
    runtime_dir = tmp_path / "watch" / "runtime"
    runtime_dir.mkdir(parents=True)

    logged_error = _persist_reload_error(
        reload_error="PyTorchStreamReader failed reading file data/0",
        runtime_dir=runtime_dir,
        last_logged_reload_error=None,
    )

    assert logged_error == "PyTorchStreamReader failed reading file data/0"
    assert (tmp_path / "watch" / "reload_error.log").read_text(encoding="utf-8") == (
        "PyTorchStreamReader failed reading file data/0\n"
    )


def _sample_telemetry(
    *,
    state_labels: tuple[str, ...] = ("active",),
) -> FZeroXTelemetry:
    return make_telemetry(
        state_labels=state_labels,
        speed_kph=0.0,
        race_distance=-3040.8,
        lap_distance=75987.2,
        race_time_ms=116,
    )
