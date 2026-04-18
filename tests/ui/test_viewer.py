# tests/ui/test_viewer.py
import os
from pathlib import Path

import numpy as np
import pygame

from fzerox_emulator import (
    JOYPAD_A,
    JOYPAD_START,
    JOYPAD_UP,
    ControllerState,
    display_size,
)
from rl_fzerox.core.envs.observations import STATE_FEATURE_NAMES, state_feature_names
from rl_fzerox.ui.watch.hud.format import (
    _format_observation_summary,
    _format_policy_action,
    _format_reload_age,
    _format_reload_error,
    _pressed_button_labels,
)
from rl_fzerox.ui.watch.hud.model import (
    _build_panel_columns,
    _observation_preview_size,
    _panel_content_height,
    _preview_frame,
    _window_size,
)
from rl_fzerox.ui.watch.render.frame import _create_fonts
from rl_fzerox.ui.watch.runtime.episode_result import _update_best_finish_position
from rl_fzerox.ui.watch.runtime.timing import (
    _adjust_control_fps,
    _resolve_control_fps,
    _resolve_render_fps,
)
from rl_fzerox.ui.watch.session import _persist_reload_error
from tests.ui.viewer_support import sample_telemetry as _sample_telemetry


def test_target_display_size_applies_aspect_correction() -> None:
    frame_shape = np.zeros((240, 640, 3), dtype=np.uint8).shape

    corrected_display_size = display_size(frame_shape, 4.0 / 3.0)

    assert corrected_display_size == (640, 480)


def test_target_display_size_falls_back_to_raw_frame_size() -> None:
    frame_shape = np.zeros((240, 640, 3), dtype=np.uint8).shape

    corrected_display_size = display_size(frame_shape, 0.0)

    assert corrected_display_size == (640, 240)


def test_window_size_adds_sidebar_width() -> None:
    assert _window_size((592, 444), (84, 116, 12)) == (1204, 880)


def test_pressed_button_labels_are_human_readable() -> None:
    assert _pressed_button_labels(0) == "none"
    assert (
        _pressed_button_labels((1 << JOYPAD_UP) | (1 << JOYPAD_A) | (1 << JOYPAD_START))
        == "Up A Start"
    )


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
            stuck_step_limit=240,
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
            <= _window_size((592, 444), (84, 116, 12))[1]
        )
    finally:
        pygame.quit()


def test_side_panel_drops_cockpit_control_section() -> None:
    columns = _build_panel_columns(
        episode=0,
        info={"frame_index": 0, "native_fps": 60.0},
        reset_info={},
        episode_reward=0.0,
        paused=False,
        control_state=ControllerState(),
        policy_curriculum_stage=None,
        policy_action=None,
        policy_reload_age_seconds=None,
        policy_reload_error=None,
        action_repeat=3,
        stuck_step_limit=240,
        stuck_min_speed_kph=50.0,
        game_display_size=(592, 444),
        observation_shape=(84, 116, 12),
        telemetry=_sample_telemetry(),
    )

    assert [section.title for section in columns.left] == ["Session"]


def test_session_section_shows_episode_step_counter() -> None:
    columns = _build_panel_columns(
        episode=0,
        info={
            "frame_index": 0,
            "native_fps": 60.0,
            "episode_step": 123,
            "reverse_timer": 45,
            "progress_frontier_stalled_frames": 300,
        },
        reset_info={},
        episode_reward=0.0,
        paused=False,
        control_state=ControllerState(),
        policy_curriculum_stage=None,
        policy_action=None,
        policy_reload_age_seconds=None,
        policy_reload_error=None,
        action_repeat=3,
        max_episode_steps=50_000,
        stuck_step_limit=240,
        wrong_way_timer_limit=120,
        progress_frontier_stall_limit_frames=900,
        stuck_min_speed_kph=50.0,
        game_display_size=(592, 444),
        observation_shape=(84, 116, 12),
        telemetry=_sample_telemetry(),
    )

    session_section = columns.left[0]
    steps_line = next(line for line in session_section.lines if line.label == "Steps")
    reverse_line = next(line for line in session_section.lines if line.label == "Reverse")
    frontier_line = next(line for line in session_section.lines if line.label == "Frontier")
    assert steps_line.value == "123 / 50000"
    assert reverse_line.value == "45 / 120"
    assert frontier_line.value == "300 / 900"


def test_session_section_marks_reverse_truncation_as_off() -> None:
    columns = _build_panel_columns(
        episode=0,
        info={"frame_index": 0, "native_fps": 60.0, "reverse_timer": 45},
        reset_info={},
        episode_reward=0.0,
        paused=False,
        control_state=ControllerState(),
        policy_curriculum_stage=None,
        policy_action=None,
        policy_reload_age_seconds=None,
        policy_reload_error=None,
        action_repeat=3,
        max_episode_steps=50_000,
        stuck_step_limit=240,
        wrong_way_timer_limit=None,
        progress_frontier_stall_limit_frames=900,
        stuck_min_speed_kph=50.0,
        game_display_size=(592, 444),
        observation_shape=(84, 116, 12),
        telemetry=_sample_telemetry(),
    )

    session_section = columns.left[0]
    reverse_line = next(line for line in session_section.lines if line.label == "Reverse")
    assert reverse_line.value == "45 / off"


def test_preview_frame_shows_stacked_rgb_observations_as_grid() -> None:
    first = np.zeros((2, 3, 3), dtype=np.uint8)
    second = np.full((2, 3, 3), 255, dtype=np.uint8)
    stacked = np.concatenate((first, second), axis=2)

    preview = _preview_frame(stacked)

    assert preview.shape == (2, 6, 3)
    assert np.array_equal(preview[:, :3, :], first)
    assert np.array_equal(preview[:, 3:, :], second)


def test_preview_frame_shows_rgb_gray_observations_as_grid() -> None:
    history = np.zeros((2, 3, 3), dtype=np.uint8)
    history[:, :, 0] = 32
    history[:, :, 1] = 96
    history[:, :, 2] = 160
    current = np.full((2, 3, 3), 255, dtype=np.uint8)
    stacked = np.concatenate((history, current), axis=2)
    info = {"observation_stack": 4, "observation_stack_mode": "rgb_gray"}

    preview = _preview_frame(stacked, info=info)

    assert preview.shape == (2, 12, 3)
    assert np.array_equal(preview[:2, :3, :], np.repeat(history[:, :, 0:1], 3, axis=2))
    assert np.array_equal(preview[:2, 3:6, :], np.repeat(history[:, :, 1:2], 3, axis=2))
    assert np.array_equal(preview[:2, 6:9, :], np.repeat(history[:, :, 2:3], 3, axis=2))
    assert np.array_equal(preview[:2, 9:12, :], current)
    assert _observation_preview_size(stacked.shape, info=info) == (12, 2)
    assert (
        _format_observation_summary(
            stacked.shape,
            info=info,
        )
        == "3x2 rgb+gray x4 strip"
    )


def test_format_policy_action_is_human_readable() -> None:
    assert _format_policy_action(None) == "manual"
    assert _format_policy_action(np.array([2, 0], dtype=np.int64)) == "[2,0]"
    assert _format_policy_action(np.array([4, 1], dtype=np.int64)) == "[4,1]"
    assert _format_policy_action(np.array([4, 1, 1, 2], dtype=np.int64)) == "[4,1,1,2]"
    assert _format_policy_action(np.array([0.25, -0.75], dtype=np.float32)) == "[+0.25,-0.75]"
    assert (
        _format_policy_action(
            {
                "continuous": np.array([0.25, 0.5], dtype=np.float32),
                "discrete": np.array([1], dtype=np.int64),
            }
        )
        == "c=[+0.25,+0.50] d=[1]"
    )


def test_display_section_includes_action_repeat() -> None:
    columns = _build_panel_columns(
        episode=0,
        info={
            "frame_index": 0,
            "native_fps": 60.0,
            "control_fps": 30.0,
            "game_fps": 60.0,
            "control_fps_target": 120.0,
            "game_fps_target": 240.0,
            "render_fps": 60.0,
            "render_fps_target": 60.0,
        },
        reset_info={},
        episode_reward=0.0,
        paused=False,
        control_state=ControllerState(),
        policy_curriculum_stage=None,
        policy_action=np.array([2, 1, 0], dtype=np.int64),
        policy_reload_age_seconds=5.0,
        policy_reload_error=None,
        action_repeat=2,
        stuck_step_limit=240,
        stuck_min_speed_kph=50.0,
        game_display_size=(592, 444),
        observation_shape=(84, 116, 12),
        telemetry=_sample_telemetry(),
    )

    display_section = next(section for section in columns.right if section.title == "Display")
    repeat_line = next(line for line in display_section.lines if line.label == "Frame skip")
    control_rate_line = next(line for line in display_section.lines if line.label == "Control FPS")
    game_rate_line = next(line for line in display_section.lines if line.label == "Game FPS")
    render_rate_line = next(line for line in display_section.lines if line.label == "Render FPS")

    assert repeat_line.value == "2"
    assert control_rate_line.value == "30.0 / 120.0"
    assert game_rate_line.value == "60.0 / 240.0"
    assert render_rate_line.value == "60.0 / 60.0"


def test_watch_fps_helpers_resolve_split_control_and_render_rates() -> None:
    assert _resolve_control_fps("auto", native_control_fps=30.0) == 30.0
    assert _resolve_control_fps("unlimited", native_control_fps=30.0) is None
    assert _resolve_control_fps(120.0, native_control_fps=30.0) == 120.0
    assert _resolve_render_fps(None, native_fps=60.0) == 60.0
    assert _resolve_render_fps("auto", native_fps=60.0) == 60.0
    assert _resolve_render_fps("unlimited", native_fps=60.0) is None


def test_watch_control_fps_adjustment_supports_uncapped_mode() -> None:
    assert _adjust_control_fps(60.0, 1, native_control_fps=60.0) == 75.0
    assert _adjust_control_fps(60.0, -1, native_control_fps=60.0) == 45.0
    assert _adjust_control_fps(None, 1, native_control_fps=60.0) is None
    assert _adjust_control_fps(None, -1, native_control_fps=60.0) == 45.0


def test_side_panel_can_show_policy_observation_state_vector() -> None:
    columns = _build_panel_columns(
        episode=0,
        info={"frame_index": 0, "native_fps": 60.0},
        reset_info={},
        episode_reward=0.0,
        paused=False,
        control_state=ControllerState(),
        policy_curriculum_stage=None,
        policy_action=np.array([2, 1, 0], dtype=np.int64),
        policy_reload_age_seconds=5.0,
        policy_reload_error=None,
        action_repeat=1,
        stuck_step_limit=240,
        stuck_min_speed_kph=50.0,
        game_display_size=(592, 444),
        observation_shape=(84, 116, 12),
        observation_state=np.array(
            [0.5, 0.75, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.2, 0.25],
            dtype=np.float32,
        ),
        observation_state_feature_names=STATE_FEATURE_NAMES,
        telemetry=_sample_telemetry(),
    )

    obs_state_section = next(section for section in columns.right if section.title == "Obs State")
    values = {line.label: line.value for line in obs_state_section.lines}

    assert values == {
        "speed_norm": "0.500",
        "energy_frac": "0.750",
        "reverse_active": "1.000",
        "airborne": "0.000",
        "can_boost": "1.000",
        "boost_active": "0.000",
        "left_lean_held": "0.000",
        "right_lean_held": "1.000",
        "left_press_age_norm": "1.000",
        "right_press_age_norm": "0.200",
        "recent_boost_pressure": "0.250",
    }


def test_side_panel_splits_observation_action_buffer_from_state() -> None:
    feature_names = state_feature_names("race_core", action_history_len=2)
    columns = _build_panel_columns(
        episode=0,
        info={"frame_index": 0, "native_fps": 60.0},
        reset_info={},
        episode_reward=0.0,
        paused=False,
        control_state=ControllerState(),
        policy_curriculum_stage=None,
        policy_action=None,
        policy_reload_age_seconds=0.0,
        policy_reload_error=None,
        action_repeat=1,
        stuck_step_limit=240,
        stuck_min_speed_kph=50.0,
        game_display_size=(592, 444),
        observation_shape=(98, 130, 9),
        observation_state=np.arange(len(feature_names), dtype=np.float32),
        observation_state_feature_names=feature_names,
        telemetry=_sample_telemetry(),
    )

    obs_state_section = next(section for section in columns.right if section.title == "Obs State")
    action_buffer_section = next(
        section for section in columns.right if section.title == "Action Buffer"
    )

    assert [line.label for line in obs_state_section.lines] == [
        "speed_norm",
        "energy_frac",
        "reverse_active",
        "airborne",
        "can_boost",
        "boost_active",
    ]
    assert [line.label for line in action_buffer_section.lines] == [
        "prev_steer_1",
        "prev_steer_2",
        "prev_gas_1",
        "prev_gas_2",
        "prev_boost_1",
        "prev_boost_2",
        "prev_lean_1",
        "prev_lean_2",
    ]


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
            stuck_step_limit=240,
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
            <= _window_size((592, 444), observation_shape)[1]
        )
    finally:
        pygame.quit()


def test_session_section_includes_stuck_counter() -> None:
    columns = _build_panel_columns(
        episode=0,
        info={"frame_index": 0, "native_fps": 60.0, "stalled_steps": 17},
        reset_info={},
        episode_reward=0.0,
        paused=False,
        control_state=ControllerState(),
        policy_curriculum_stage="lean_enabled",
        policy_action=np.array([2, 1, 0], dtype=np.int64),
        policy_reload_age_seconds=5.0,
        policy_reload_error=None,
        action_repeat=1,
        stuck_step_limit=240,
        stuck_min_speed_kph=50.0,
        game_display_size=(592, 444),
        observation_shape=(84, 116, 12),
        telemetry=_sample_telemetry(),
    )

    session_section = next(section for section in columns.left if section.title == "Session")
    stuck_line = next(line for line in session_section.lines if line.label == "Stuck")

    assert stuck_line.value == "17 / 240"


def test_session_section_shows_curriculum_stage_name() -> None:
    columns = _build_panel_columns(
        episode=0,
        info={
            "frame_index": 0,
            "native_fps": 60.0,
        },
        reset_info={},
        episode_reward=0.0,
        paused=False,
        control_state=ControllerState(),
        policy_curriculum_stage="lean_enabled",
        policy_action=np.array([2, 1, 0], dtype=np.int64),
        policy_reload_age_seconds=5.0,
        policy_reload_error=None,
        action_repeat=1,
        stuck_step_limit=240,
        stuck_min_speed_kph=50.0,
        game_display_size=(592, 444),
        observation_shape=(84, 116, 12),
        telemetry=_sample_telemetry(),
    )

    session_section = next(section for section in columns.left if section.title == "Session")
    curriculum_line = next(
        line for line in session_section.lines if line.label == "Checkpoint stage"
    )

    assert curriculum_line.value == "lean_enabled"


def test_session_section_shows_policy_deterministic_mode() -> None:
    columns = _build_panel_columns(
        episode=0,
        info={"frame_index": 0, "native_fps": 60.0},
        reset_info={},
        episode_reward=0.0,
        paused=False,
        control_state=ControllerState(),
        policy_curriculum_stage=None,
        policy_deterministic=False,
        policy_action=np.array([2, 1, 0], dtype=np.int64),
        policy_reload_age_seconds=5.0,
        policy_reload_error=None,
        action_repeat=1,
        stuck_step_limit=240,
        stuck_min_speed_kph=50.0,
        game_display_size=(592, 444),
        observation_shape=(84, 116, 12),
        telemetry=_sample_telemetry(),
    )

    session_section = next(section for section in columns.left if section.title == "Session")
    deterministic_line = next(
        line for line in session_section.lines if line.label == "Deterministic"
    )

    assert deterministic_line.value == "false"


def test_session_section_formats_hybrid_action_value_with_fixed_digits() -> None:
    columns = _build_panel_columns(
        episode=0,
        info={"frame_index": 0, "native_fps": 60.0},
        reset_info={},
        episode_reward=0.0,
        paused=False,
        control_state=ControllerState(),
        policy_curriculum_stage=None,
        policy_deterministic=False,
        policy_action={
            "continuous": np.array([0.0, 0.5, -0.5], dtype=np.float32),
            "discrete": np.array([0, 0], dtype=np.int64),
        },
        policy_reload_age_seconds=5.0,
        policy_reload_error=None,
        action_repeat=1,
        stuck_step_limit=240,
        stuck_min_speed_kph=50.0,
        game_display_size=(592, 444),
        observation_shape=(84, 116, 12),
        telemetry=_sample_telemetry(),
    )

    session_section = next(section for section in columns.left if section.title == "Session")
    action_line = next(line for line in session_section.lines if line.label == "Action")

    assert action_line.value == "c=[+0.00,+0.50,-0.50] d=[0,0]"


def test_session_section_shows_reward_with_four_decimals() -> None:
    columns = _build_panel_columns(
        episode=0,
        info={
            "frame_index": 0,
            "native_fps": 60.0,
            "step_reward": -0.016,
        },
        reset_info={},
        episode_reward=-12.34567,
        paused=False,
        control_state=ControllerState(),
        policy_curriculum_stage=None,
        policy_action=np.array([2, 1, 0], dtype=np.int64),
        policy_reload_age_seconds=5.0,
        policy_reload_error=None,
        action_repeat=1,
        stuck_step_limit=240,
        stuck_min_speed_kph=50.0,
        game_display_size=(592, 444),
        observation_shape=(84, 116, 12),
        telemetry=_sample_telemetry(),
    )

    session_section = next(section for section in columns.left if section.title == "Session")
    step_line = next(line for line in session_section.lines if line.label == "Step")
    return_line = next(line for line in session_section.lines if line.label == "Return")

    assert step_line.value == "-0.0160"
    assert return_line.value == "-12.3457"


def test_session_section_shows_best_finish_position() -> None:
    columns = _build_panel_columns(
        episode=2,
        info={"frame_index": 0, "native_fps": 60.0},
        reset_info={},
        episode_reward=0.0,
        paused=False,
        control_state=ControllerState(),
        policy_curriculum_stage=None,
        policy_action=np.array([2, 1, 0], dtype=np.int64),
        policy_reload_age_seconds=5.0,
        policy_reload_error=None,
        best_finish_position=8,
        action_repeat=1,
        stuck_step_limit=240,
        stuck_min_speed_kph=50.0,
        game_display_size=(592, 444),
        observation_shape=(84, 116, 12),
        telemetry=_sample_telemetry(),
    )

    session_section = next(section for section in columns.left if section.title == "Session")
    best_position_line = next(
        line for line in session_section.lines if line.label == "Best position"
    )

    assert best_position_line.value == "8"


def test_session_section_shows_na_before_successful_finish() -> None:
    columns = _build_panel_columns(
        episode=2,
        info={"frame_index": 0, "native_fps": 60.0},
        reset_info={},
        episode_reward=0.0,
        paused=False,
        control_state=ControllerState(),
        policy_curriculum_stage=None,
        policy_action=np.array([2, 1, 0], dtype=np.int64),
        policy_reload_age_seconds=5.0,
        policy_reload_error=None,
        action_repeat=1,
        stuck_step_limit=240,
        stuck_min_speed_kph=50.0,
        game_display_size=(592, 444),
        observation_shape=(84, 116, 12),
        telemetry=_sample_telemetry(),
    )

    session_section = next(section for section in columns.left if section.title == "Session")
    best_position_line = next(
        line for line in session_section.lines if line.label == "Best position"
    )

    assert best_position_line.value == "n/a"


def test_best_finish_position_tracks_only_finished_episodes() -> None:
    best_position = _update_best_finish_position(
        None,
        {"termination_reason": "crashed", "position": 4},
        _sample_telemetry(position=4),
    )
    assert best_position is None

    best_position = _update_best_finish_position(
        best_position,
        {"termination_reason": "finished"},
        _sample_telemetry(position=8),
    )
    assert best_position == 8

    best_position = _update_best_finish_position(
        best_position,
        {"termination_reason": "finished"},
        _sample_telemetry(position=12),
    )
    assert best_position == 8

    best_position = _update_best_finish_position(
        best_position,
        {"termination_reason": "finished"},
        _sample_telemetry(position=3),
    )
    assert best_position == 3


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


def test_persist_reload_error_writes_full_message_once(tmp_path: Path) -> None:
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
