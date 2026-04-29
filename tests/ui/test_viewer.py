# tests/ui/test_viewer.py
from pathlib import Path

import numpy as np

from fzerox_emulator import (
    JOYPAD_A,
    JOYPAD_START,
    JOYPAD_UP,
    ControllerState,
)
from rl_fzerox.core.config.schema import (
    PolicyConfig,
    TrainConfig,
)
from rl_fzerox.core.envs.observations import (
    DEFAULT_STATE_VECTOR_SPEC,
    state_feature_names,
)
from rl_fzerox.ui.watch.view.components.macro_legend import (
    MACRO_LEGEND_HINTS,
    _macro_legend_rows,
)
from rl_fzerox.ui.watch.view.panels.core.format import (
    _format_policy_action,
    _format_reload_age,
    _pressed_button_labels,
)
from rl_fzerox.ui.watch.view.panels.core.model import _build_panel_columns
from rl_fzerox.ui.watch.view.screen.frame import (
    _game_course_overlay_label,
    _game_speed_overlay_label,
)
from tests.ui.viewer_support import (
    fake_viewer_fonts,
)
from tests.ui.viewer_support import (
    panel_group_labels as _panel_group_labels,
)
from tests.ui.viewer_support import (
    panel_group_values as _panel_group_values,
)
from tests.ui.viewer_support import (
    sample_telemetry as _sample_telemetry,
)


def test_pressed_button_labels_are_human_readable() -> None:
    assert _pressed_button_labels(0) == "none"
    assert (
        _pressed_button_labels((1 << JOYPAD_UP) | (1 << JOYPAD_A) | (1 << JOYPAD_START))
        == "Up A Start"
    )


def test_game_course_overlay_label_prefers_cup_and_course_name() -> None:
    assert (
        _game_course_overlay_label(
            {
                "track_course_ref": "joker/big_hand",
                "track_course_name": "Big Hand",
            }
        )
        == "Joker Cup : Big Hand"
    )


def test_game_speed_overlay_label_formats_actual_speedup() -> None:
    assert (
        _game_speed_overlay_label(
            {"native_fps": 60.0, "game_fps": 120.0},
            action_repeat=2,
        )
        == "2.0x"
    )
    assert (
        _game_speed_overlay_label(
            {"native_fps": 60.0, "game_fps": 150.0},
            action_repeat=2,
        )
        == "2.5x"
    )
    assert (
        _game_speed_overlay_label(
            {"native_fps": 60.0, "control_fps": 45.0},
            action_repeat=2,
        )
        == "1.5x"
    )
    assert (
        _game_speed_overlay_label(
            {"native_fps": 60.0, "game_fps": 720.0},
            action_repeat=2,
        )
        == "12.0x"
    )


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
        stuck_min_speed_kph=50.0,
        game_display_size=(592, 444),
        observation_shape=(84, 116, 12),
        telemetry=_sample_telemetry(),
    )

    assert "Cockpit Control" not in [section.title for section in columns.left]
    assert [section.title for section in columns.left] == [
        "Run",
        "Policy Details",
        "Race",
        "Game Details",
        "Timing",
        "Display",
    ]
    assert [section.title for section in columns.middle] == [
        "Track Geometry",
    ]


def test_train_tab_shows_current_training_hparams() -> None:
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
        stuck_min_speed_kph=50.0,
        game_display_size=(592, 444),
        observation_shape=(84, 116, 12),
        telemetry=_sample_telemetry(),
        train_config=TrainConfig(
            algorithm="maskable_hybrid_action_ppo",
            total_timesteps=50_000_000,
            learning_rate=2e-4,
            n_steps=2048,
            output_root=Path("local/runs"),
            run_name="wide_ppo",
        ),
        policy_config=PolicyConfig.model_validate({"extractor": {"conv_profile": "nature_wide"}}),
    )

    training_section = next(section for section in columns.train if section.title == "Training")
    policy_section = next(section for section in columns.train if section.title == "Policy")
    training_values = {line.label: line.value for line in training_section.lines}
    policy_values = {line.label: line.value for line in policy_section.lines}

    assert training_values["Algorithm"] == "maskable_hybrid_action_ppo"
    assert "Run" not in training_values
    assert training_values["Target steps"] == "50,000,000"
    assert training_values["LR"] == "2e-04"
    assert policy_values["Conv"] == "nature_wide"
    assert policy_values["Pi net"] == "[256, 256]"


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
        progress_frontier_stall_limit_frames=900,
        stuck_min_speed_kph=50.0,
        game_display_size=(592, 444),
        observation_shape=(84, 116, 12),
        telemetry=_sample_telemetry(),
    )

    session_section = columns.left[0]
    episode_frame_line = next(
        line for line in session_section.lines if line.label == "Episode frame"
    )
    env_step_line = next(line for line in session_section.lines if line.label == "Env step")
    frontier_line = next(line for line in session_section.lines if line.label == "Frontier frames")
    assert episode_frame_line.value == "123 / 50000"
    assert env_step_line.value == "41 / 16667"
    assert frontier_line.value == "300 / 900"


def test_session_section_omits_reverse_and_stuck_counters() -> None:
    columns = _build_panel_columns(
        episode=0,
        info={
            "frame_index": 0,
            "native_fps": 60.0,
            "reverse_timer": 45,
            "stalled_steps": 17,
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
        progress_frontier_stall_limit_frames=900,
        stuck_min_speed_kph=50.0,
        game_display_size=(592, 444),
        observation_shape=(84, 116, 12),
        telemetry=_sample_telemetry(),
    )

    session_section = columns.left[0]
    labels = {line.label for line in session_section.lines}
    assert "Reverse frames" not in labels
    assert "Stuck frames" not in labels

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
        stuck_min_speed_kph=50.0,
        game_display_size=(592, 444),
        observation_shape=(84, 116, 12),
        telemetry=_sample_telemetry(),
    )

    timing_section = next(section for section in columns.left if section.title == "Timing")
    repeat_line = next(line for line in timing_section.lines if line.label == "Action repeat")
    control_rate_line = next(line for line in timing_section.lines if line.label == "Control FPS")
    speed_line = next(line for line in timing_section.lines if line.label == "Game speed")
    game_rate_line = next(line for line in timing_section.lines if line.label == "Game FPS")
    render_rate_line = next(line for line in timing_section.lines if line.label == "Render FPS")

    assert repeat_line.value == "2"
    assert control_rate_line.value == "30.0 / 120.0"
    assert speed_line.value == "1.00x"
    assert game_rate_line.value == "60.0 / 240.0"
    assert render_rate_line.value == "60.0 / 60.0"


def test_timing_section_shows_game_speed_multiplier() -> None:
    columns = _build_panel_columns(
        episode=0,
        info={
            "frame_index": 0,
            "native_fps": 60.0,
            "control_fps": 60.0,
            "control_fps_target": 60.0,
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
        stuck_min_speed_kph=50.0,
        game_display_size=(592, 444),
        observation_shape=(84, 116, 12),
        telemetry=_sample_telemetry(),
    )

    timing_section = next(section for section in columns.left if section.title == "Timing")
    speed_line = next(line for line in timing_section.lines if line.label == "Game speed")

    assert speed_line.value == "2.00x"


def test_macro_legend_replaces_side_panel_key_lines() -> None:
    columns = _build_panel_columns(
        episode=0,
        info={},
        reset_info={},
        episode_reward=0.0,
        paused=False,
        control_state=ControllerState(),
        policy_curriculum_stage=None,
        policy_action=np.array([2, 1, 0], dtype=np.int64),
        policy_reload_age_seconds=5.0,
        policy_reload_error=None,
        action_repeat=2,
        stuck_min_speed_kph=50.0,
        game_display_size=(592, 444),
        observation_shape=(84, 116, 12),
        telemetry=_sample_telemetry(),
    )

    display_section = next(section for section in columns.left if section.title == "Display")
    key_map = {line.label: line.value for line in display_section.lines}
    hint_map = {hint.keys: (hint.controller, hint.action) for hint in MACRO_LEGEND_HINTS}

    assert "Keys" not in key_map
    assert "More keys" not in key_map
    assert hint_map == {
        "Esc": (None, "close"),
        "P": (None, "pause"),
        "N": (None, "step"),
        "R": (None, "reset"),
        "K": (None, "save"),
        "M": (None, "manual"),
        "D": (None, "policy"),
        "Tab / 1-6": (None, "tabs"),
        "+/-": (None, "speed"),
        "0": (None, "realtime"),
        "Arrow keys": ("stick X/Y", "steer/pitch"),
        "Z": ("A", "accelerate"),
        "X": ("C-down", "air brake"),
        "Space": ("B", "boost"),
        "A": ("Z", "lean left"),
        "S": ("R", "lean right"),
        "Enter": ("Start", "start"),
    }


def test_macro_legend_wraps_inside_preview_column() -> None:
    fonts = fake_viewer_fonts()
    rows = _macro_legend_rows(font=fonts.small, width=568)

    assert len(rows) > 1
    assert tuple(hint for row in rows for hint in row) == MACRO_LEGEND_HINTS


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
        stuck_min_speed_kph=50.0,
        game_display_size=(592, 444),
        observation_shape=(84, 116, 12),
        observation_state=np.array(
            [0.5, 0.75, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.2, 0.25],
            dtype=np.float32,
        ),
        observation_state_feature_names=DEFAULT_STATE_VECTOR_SPEC.names,
        telemetry=_sample_telemetry(),
    )

    state_vector_section = next(
        section for section in columns.stats if section.title == "State Vector"
    )
    values = _panel_group_values(state_vector_section, "State")

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


def test_side_panel_splits_profile_action_history_from_state_vector() -> None:
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
        stuck_min_speed_kph=50.0,
        game_display_size=(592, 444),
        observation_shape=(98, 130, 9),
        observation_state=np.arange(len(feature_names), dtype=np.float32),
        observation_state_feature_names=feature_names,
        telemetry=_sample_telemetry(),
    )

    state_vector_section = next(
        section for section in columns.stats if section.title == "State Vector"
    )

    assert _panel_group_labels(state_vector_section, "State") == [
        "speed_norm",
        "energy_frac",
        "reverse_active",
        "airborne",
        "can_boost",
        "boost_active",
    ]
    assert _panel_group_labels(state_vector_section, "Control History") == [
        "prev_steer_1",
        "prev_steer_2",
        "prev_gas_1",
        "prev_gas_2",
        "prev_boost_1",
        "prev_boost_2",
        "prev_lean_1",
        "prev_lean_2",
    ]


def test_side_panel_groups_component_state_vector_by_component() -> None:
    feature_names = (
        "vehicle_state.speed_norm",
        "machine_context.engine",
        "surface_state.on_dirt_surface",
        "course_context.course_builtin_00",
        "course_context.course_builtin_01",
        "course_context.course_builtin_02",
        "control_history.prev_steer_1",
        "control_history.prev_thrust_1",
    )
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
        stuck_min_speed_kph=50.0,
        game_display_size=(592, 444),
        observation_shape=(98, 130, 9),
        observation_state=np.array(
            [0.5, 0.7, 1.0, 0.0, 1.0, 0.0, -1.0, 1.0],
            dtype=np.float32,
        ),
        observation_state_feature_names=feature_names,
        telemetry=_sample_telemetry(),
    )

    state_vector_section = next(
        section for section in columns.stats if section.title == "State Vector"
    )

    assert _panel_group_labels(state_vector_section, "Vehicle") == ["speed_norm"]
    assert _panel_group_labels(state_vector_section, "Machine") == ["engine"]
    assert _panel_group_labels(state_vector_section, "Surface") == ["on_dirt_surface"]
    assert _panel_group_values(state_vector_section, "Course") == {"course": "1 | 010"}
    assert _panel_group_labels(state_vector_section, "Control History") == [
        "prev_steer_1",
        "prev_thrust_1",
    ]


def test_side_panel_marks_zeroed_state_features_inside_track_position() -> None:
    feature_names = (
        "vehicle_state.speed_norm",
        "track_position.lap_progress",
        "track_position.edge_ratio",
        "track_position.outside_track_bounds",
    )
    columns = _build_panel_columns(
        episode=0,
        info={
            "frame_index": 0,
            "native_fps": 60.0,
            "observation_zeroed_state_features": (
                "track_position.edge_ratio",
                "track_position.outside_track_bounds",
            ),
        },
        reset_info={},
        episode_reward=0.0,
        paused=False,
        control_state=ControllerState(),
        policy_curriculum_stage=None,
        policy_action=None,
        policy_reload_age_seconds=0.0,
        policy_reload_error=None,
        action_repeat=1,
        stuck_min_speed_kph=50.0,
        game_display_size=(592, 444),
        observation_shape=(60, 76, 12),
        observation_state=np.array([0.5, 0.25, 0.0, 0.0], dtype=np.float32),
        observation_state_feature_names=feature_names,
        telemetry=_sample_telemetry(),
    )

    state_vector_section = next(
        section for section in columns.stats if section.title == "State Vector"
    )

    assert _panel_group_labels(state_vector_section, "Track Position") == [
        "lap_progress",
        "// edge_ratio",
        "// outside_track_bounds",
    ]


def test_side_panel_marks_zeroed_state_components() -> None:
    feature_names = (
        "vehicle_state.speed_norm",
        "track_position.lap_progress",
        "track_position.edge_ratio",
    )
    columns = _build_panel_columns(
        episode=0,
        info={
            "frame_index": 0,
            "native_fps": 60.0,
            "observation_zeroed_state_components": ("track_position",),
        },
        reset_info={},
        episode_reward=0.0,
        paused=False,
        control_state=ControllerState(),
        policy_curriculum_stage=None,
        policy_action=None,
        policy_reload_age_seconds=0.0,
        policy_reload_error=None,
        action_repeat=1,
        stuck_min_speed_kph=50.0,
        game_display_size=(592, 444),
        observation_shape=(98, 130, 9),
        observation_state=np.array([0.5, 0.0, 0.0], dtype=np.float32),
        observation_state_feature_names=feature_names,
        telemetry=_sample_telemetry(),
    )

    state_vector_section = next(
        section for section in columns.stats if section.title == "State Vector"
    )

    assert _panel_group_labels(state_vector_section, "// Track Position") == [
        "// lap_progress",
        "// edge_ratio",
    ]

def test_session_section_shows_canonical_curriculum_stage_name() -> None:
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
        stuck_min_speed_kph=50.0,
        game_display_size=(592, 444),
        observation_shape=(84, 116, 12),
        telemetry=_sample_telemetry(),
    )

    session_section = next(section for section in columns.left if section.title == "Run")
    curriculum_line = next(line for line in session_section.lines if line.label == "Stage")

    assert curriculum_line.value == "lean_enabled"


def test_session_section_shows_checkpoint_experience_from_timesteps() -> None:
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
        policy_num_timesteps=660_000,
        policy_action=np.array([2, 1, 0], dtype=np.int64),
        policy_reload_age_seconds=5.0,
        policy_reload_error=None,
        action_repeat=2,
        stuck_min_speed_kph=50.0,
        game_display_size=(592, 444),
        observation_shape=(84, 116, 12),
        telemetry=_sample_telemetry(),
    )

    session_section = next(section for section in columns.left if section.title == "Run")
    experience_line = next(line for line in session_section.lines if line.label == "Experience")

    assert experience_line.value == "6h 06m"


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
        stuck_min_speed_kph=50.0,
        game_display_size=(592, 444),
        observation_shape=(84, 116, 12),
        telemetry=_sample_telemetry(),
    )

    session_section = next(section for section in columns.left if section.title == "Policy Details")
    deterministic_line = next(
        line for line in session_section.lines if line.label == "Deterministic"
    )

    assert deterministic_line.value == "stochastic"


def test_session_section_shows_manual_driver_mode() -> None:
    columns = _build_panel_columns(
        episode=0,
        info={"frame_index": 0, "native_fps": 60.0},
        reset_info={},
        episode_reward=0.0,
        paused=False,
        control_state=ControllerState(),
        policy_label="latest",
        policy_curriculum_stage=None,
        policy_deterministic=False,
        manual_control_enabled=True,
        policy_action=None,
        policy_reload_age_seconds=5.0,
        policy_reload_error=None,
        action_repeat=1,
        stuck_min_speed_kph=50.0,
        game_display_size=(592, 444),
        observation_shape=(84, 116, 12),
        telemetry=_sample_telemetry(),
    )

    session_section = next(section for section in columns.left if section.title == "Run")
    driver_line = next(line for line in session_section.lines if line.label == "Driver")

    assert driver_line.value == "manual"


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
        stuck_min_speed_kph=50.0,
        game_display_size=(592, 444),
        observation_shape=(84, 116, 12),
        telemetry=_sample_telemetry(),
    )

    session_section = next(section for section in columns.left if section.title == "Policy Details")
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
        stuck_min_speed_kph=50.0,
        game_display_size=(592, 444),
        observation_shape=(84, 116, 12),
        telemetry=_sample_telemetry(),
    )

    session_section = next(section for section in columns.left if section.title == "Run")
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
        stuck_min_speed_kph=50.0,
        game_display_size=(592, 444),
        observation_shape=(84, 116, 12),
        telemetry=_sample_telemetry(),
    )

    session_section = next(section for section in columns.left if section.title == "Run")
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
        stuck_min_speed_kph=50.0,
        game_display_size=(592, 444),
        observation_shape=(84, 116, 12),
        telemetry=_sample_telemetry(),
    )

    session_section = next(section for section in columns.left if section.title == "Run")
    best_position_line = next(
        line for line in session_section.lines if line.label == "Best position"
    )

    assert best_position_line.value == "n/a"

def test_format_reload_age_is_human_readable() -> None:
    assert _format_reload_age(None) == "manual"
    assert _format_reload_age(12.7) == "12s ago"
    assert _format_reload_age(125.0) == "2m 05s"
    assert _format_reload_age(3665.0) == "1h 01m"
