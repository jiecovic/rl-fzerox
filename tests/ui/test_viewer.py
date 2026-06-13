# tests/ui/test_viewer.py
from pathlib import Path

import numpy as np

from fzerox_emulator import RaceControlState
from rl_fzerox.core.runtime_spec.schema import (
    PolicyConfig,
    TrainConfig,
)
from rl_fzerox.ui.watch.view.components.macro_legend import (
    MACRO_LEGEND_HINTS,
    _macro_legend_rows,
)
from rl_fzerox.ui.watch.view.panels.content.state_vector import policy_state_sections
from rl_fzerox.ui.watch.view.panels.core.format import (
    _format_policy_action,
    _format_reload_age,
)
from rl_fzerox.ui.watch.view.panels.core.model import _build_panel_columns
from rl_fzerox.ui.watch.view.screen.frame import (
    _game_course_overlay_label,
    _game_speed_overlay_label,
    _game_status_overlay_label,
)
from rl_fzerox.ui.watch.view.screen.render import _with_viewer_rates
from tests.ui.viewer_support import (
    fake_viewer_fonts,
    record_book,
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


def test_macro_legend_includes_toggle_anchor_hotkey() -> None:
    hints = {hint.keys: hint.action for hint in MACRO_LEGEND_HINTS}

    assert hints["F"] == "toggle anchor"
    assert hints["C"] == "cnn mode"
    assert hints["Tab / 1-8"] == "tabs"


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


def test_game_course_overlay_label_marks_locked_course_lightly() -> None:
    assert (
        _game_course_overlay_label(
            {
                "track_course_id": "big_hand",
                "track_course_ref": "joker/big_hand",
                "track_course_name": "Big Hand",
            },
            reset_info={"track_sampling_locked_course_id": "big_hand"},
        )
        == "> Joker Cup : Big Hand <"
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


def test_game_status_overlay_label_uses_watch_save_notice() -> None:
    assert _game_status_overlay_label({"watch_save_notice": "alt baseline saved"}) == (
        "alt baseline saved"
    )
    assert _game_status_overlay_label({"watch_save_notice": "   "}) is None


def test_viewer_rates_preserve_runtime_game_fps() -> None:
    info = _with_viewer_rates(
        {"game_fps": 72.0},
        native_fps=60.0,
        action_repeat=2,
        current_control_fps=60.0,
        current_render_fps=60.0,
        target_control_fps=60.0,
        target_render_fps=60.0,
    )

    assert info["game_fps"] == 72.0
    assert _game_speed_overlay_label(info, action_repeat=2) == "1.2x"


def test_policy_state_sections_expose_watch_ablation_toggles() -> None:
    sections = policy_state_sections(
        observation_state=np.asarray([0.5, 1.0, 0.0], dtype=np.float32),
        observation_state_reference=np.asarray([0.5, 1.0, 0.0], dtype=np.float32),
        feature_names=(
            "vehicle_state.speed_kph_norm",
            "machine_context.energy_norm",
            "course_context.course_builtin_0",
        ),
        zeroed_features=frozenset({"machine_context.energy_norm", "course_context"}),
        watch_zeroed_features=frozenset({"vehicle_state.speed_kph_norm", "course_context"}),
    )

    lines = sections[0].lines
    speed_line = next(line for line in lines if line.label == "speed_kph_norm")
    energy_line = next(line for line in lines if line.label == "// energy_norm")
    course_line = next(line for line in lines if line.label == "// course")

    assert speed_line.status_icon == "toggle_off"
    assert speed_line.click_state_feature_name == "vehicle_state.speed_kph_norm"
    assert energy_line.status_icon == "toggle_on"
    assert energy_line.click_state_feature_name == "machine_context.energy_norm"
    assert course_line.status_icon == "toggle_off"
    assert course_line.click_state_feature_name == "course_context"


def test_side_panel_drops_cockpit_control_section() -> None:
    columns = _build_panel_columns(
        episode=0,
        info={"frame_index": 0, "native_fps": 60.0},
        reset_info={},
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

    assert "Cockpit Control" not in [section.title for section in columns.left]
    assert [section.title for section in columns.left] == [
        "Run State",
        "Episode Progress",
        "Policy Output",
        "Race State",
        "Race Setup",
        "Runtime",
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
        emulator_renderer="gliden64",
        watch_device="cpu",
        train_config=TrainConfig(
            algorithm="maskable_hybrid_action_ppo",
            device="cuda",
            total_timesteps=50_000_000,
            learning_rate=2e-4,
            n_steps=2048,
            output_root=Path("local/runs"),
            run_name="wide_ppo",
        ),
        policy_config=PolicyConfig.model_validate({"extractor": {"conv_profile": "nature"}}),
    )

    live_policy_section = next(
        section for section in columns.left if section.title == "Policy Output"
    )
    runtime_section = next(section for section in columns.left if section.title == "Runtime")
    training_section = next(section for section in columns.train if section.title == "Training")
    policy_section = next(section for section in columns.train if section.title == "Policy")
    live_policy_values = {line.label: line.value for line in live_policy_section.lines}
    runtime_values = {line.label: line.value for line in runtime_section.lines}
    training_values = {line.label: line.value for line in training_section.lines}
    policy_values = {line.label: line.value for line in policy_section.lines}

    assert training_values["Algorithm"] == "maskable_hybrid_action_ppo"
    assert training_values["Train device"] == "cuda"
    assert "Renderer" not in training_values
    assert "Device" not in training_values
    assert "Run" not in training_values
    assert training_values["Target steps"] == "50,000,000"
    assert training_values["LR"] == "2e-04"
    assert live_policy_values["Device"] == "cpu"
    assert runtime_values["Renderer"] == "gliden64"
    assert policy_values["Conv"] == "nature"
    assert policy_values["Pi net"] == "[256, 256]"


def test_state_tab_shows_auxiliary_state_predictions_next_to_targets() -> None:
    columns = _build_panel_columns(
        episode=0,
        info={"frame_index": 0, "native_fps": 60.0},
        reset_info={},
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
        policy_config=PolicyConfig.model_validate(
            {
                "auxiliary_state": {
                    "enabled": True,
                    "losses": [
                        {
                            "name": "track_position.edge_ratio",
                            "weight": 0.5,
                            "grounded_only": True,
                        },
                        {
                            "name": "surface_state.on_refill_surface",
                            "weight": 0.25,
                            "grounded_only": False,
                        },
                        {
                            "name": "course_context.builtin_course_id",
                            "weight": 1.0,
                            "grounded_only": False,
                        },
                    ],
                }
            }
        ),
        policy_auxiliary_state_predictions={
            "track_position.edge_ratio": 0.125,
            "surface_state.on_refill_surface": 0.26,
            "course_context.builtin_course_id": {"index": 7, "confidence": 0.82},
        },
        policy_auxiliary_state_targets={
            "track_position.edge_ratio": 0.25,
            "surface_state.on_refill_surface": 0.0,
            "course_context.builtin_course_id": {"index": 3},
        },
    )

    state_section = next(section for section in columns.stats if section.title == "State Vector")
    edge_ratio_line = next(
        line for line in state_section.lines if line.label == "edge_ratio (ground)"
    )
    refill_line = next(line for line in state_section.lines if line.label == "on_refill_surface")
    course_line = next(line for line in state_section.lines if line.label == "course id")

    assert edge_ratio_line.value == "  0.12 |   0.25 |    6% |       "
    assert refill_line.value == "  0.26 |   0.00 |  hit  |       "
    assert course_line.value == " 7@82% |      3 | miss  |       "


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
        control_state=RaceControlState(),
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

    session_section = next(
        section for section in columns.left if section.title == "Episode Progress"
    )
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
        control_state=RaceControlState(),
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

    session_section = next(
        section for section in columns.left if section.title == "Episode Progress"
    )
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


def test_timing_section_shows_compact_runtime_rates() -> None:
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
        control_state=RaceControlState(),
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

    runtime_section = next(section for section in columns.left if section.title == "Runtime")
    values = {line.label: line.value for line in runtime_section.lines}

    assert values == {
        "Renderer": "unknown",
        "Repeat": "x2",
        "Control FPS": "30.0",
        "Game FPS": "60.0",
        "Render FPS": "60.0",
        "Speed multiplier": "1.00x",
        "Game size": "444x592",
        "Obs size": "84x116x12",
    }


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
        control_state=RaceControlState(),
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

    runtime_section = next(section for section in columns.left if section.title == "Runtime")
    speed_line = next(line for line in runtime_section.lines if line.label == "Speed multiplier")

    assert speed_line.value == "2.00x"


def test_macro_legend_replaces_side_panel_key_lines() -> None:
    columns = _build_panel_columns(
        episode=0,
        info={},
        reset_info={},
        episode_reward=0.0,
        paused=False,
        control_state=RaceControlState(),
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

    runtime_section = next(section for section in columns.left if section.title == "Runtime")
    key_map = {line.label: line.value for line in runtime_section.lines}
    hint_map = {hint.keys: (hint.controller, hint.action) for hint in MACRO_LEGEND_HINTS}

    assert "Keys" not in key_map
    assert "More keys" not in key_map
    assert key_map["Game size"] == "444x592"
    assert key_map["Obs size"] == "84x116x12"
    assert hint_map == {
        "Esc": (None, "close"),
        "P": (None, "pause"),
        "N": (None, "step"),
        "R": (None, "same course"),
        "E": (None, "prev course"),
        "T": (None, "next course"),
        "G": (None, "difficulty"),
        "F": (None, "toggle anchor"),
        "K": (None, "save"),
        "M": (None, "manual"),
        "D": (None, "policy"),
        "Tab / 1-8": (None, "tabs"),
        "C": (None, "cnn mode"),
        "+/-": (None, "speed"),
        "0": (None, "reset speed"),
        "Arrow keys": ("stick X/Y", "steer/pitch"),
        "Z": ("A", "accelerate"),
        "X": ("C-down", "air brake"),
        "Space": ("B", "boost"),
        "A": ("Z", "lean left"),
        "S": ("R", "lean right"),
        "Q": (None, "spin left"),
        "W": (None, "spin right"),
        "Enter": ("Start", "start"),
    }


def test_macro_legend_wraps_inside_preview_column() -> None:
    fonts = fake_viewer_fonts()
    rows = _macro_legend_rows(font=fonts.small, width=568)

    assert len(rows) > 1
    assert tuple(hint for row in rows for hint in row) == MACRO_LEGEND_HINTS


def test_side_panel_can_show_policy_observation_state_vector() -> None:
    feature_names = (
        "vehicle_state.speed_norm",
        "vehicle_state.energy_frac",
        "vehicle_state.reverse_active",
        "vehicle_state.airborne",
        "vehicle_state.boost_unlocked",
        "vehicle_state.boost_active",
        "vehicle_state.lateral_velocity_norm",
        "vehicle_state.sliding_active",
    )
    columns = _build_panel_columns(
        episode=0,
        info={"frame_index": 0, "native_fps": 60.0},
        reset_info={},
        episode_reward=0.0,
        paused=False,
        control_state=RaceControlState(),
        policy_curriculum_stage=None,
        policy_action=np.array([2, 1, 0], dtype=np.int64),
        policy_reload_age_seconds=5.0,
        policy_reload_error=None,
        action_repeat=1,
        stuck_min_speed_kph=50.0,
        game_display_size=(592, 444),
        observation_shape=(84, 116, 12),
        observation_state=np.array(
            [0.5, 0.75, 1.0, 0.0, 1.0, 0.0, 0.25, 1.0],
            dtype=np.float32,
        ),
        observation_state_feature_names=feature_names,
        telemetry=_sample_telemetry(),
    )

    state_vector_section = next(
        section for section in columns.stats if section.title == "State Vector"
    )
    values = _panel_group_values(state_vector_section, "Vehicle")

    assert values == {
        "speed_norm": "0.500",
        "energy_frac": "0.750",
        "reverse_active": "1.000",
        "airborne": "0.000",
        "boost_unlocked": "1.000",
        "boost_active": "0.000",
        "lateral_velocity_norm": "0.250",
        "sliding_active": "1.000",
    }


def test_side_panel_splits_component_action_history_from_state_vector() -> None:
    feature_names = (
        "vehicle_state.speed_norm",
        "vehicle_state.energy_frac",
        "vehicle_state.reverse_active",
        "vehicle_state.airborne",
        "vehicle_state.boost_unlocked",
        "vehicle_state.boost_active",
        "control_history.prev_steer_1",
        "control_history.prev_steer_2",
        "control_history.prev_thrust_1",
        "control_history.prev_thrust_2",
        "control_history.prev_boost_1",
        "control_history.prev_boost_2",
        "control_history.prev_lean_1",
        "control_history.prev_lean_2",
    )
    columns = _build_panel_columns(
        episode=0,
        info={"frame_index": 0, "native_fps": 60.0},
        reset_info={},
        episode_reward=0.0,
        paused=False,
        control_state=RaceControlState(),
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

    assert _panel_group_labels(state_vector_section, "Vehicle") == [
        "speed_norm",
        "energy_frac",
        "reverse_active",
        "airborne",
        "boost_unlocked",
        "boost_active",
    ]
    assert _panel_group_labels(state_vector_section, "Control History") == [
        "prev_steer_1",
        "prev_steer_2",
        "prev_thrust_1",
        "prev_thrust_2",
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
        control_state=RaceControlState(),
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
        control_state=RaceControlState(),
        policy_curriculum_stage=None,
        policy_action=None,
        policy_reload_age_seconds=0.0,
        policy_reload_error=None,
        action_repeat=1,
        stuck_min_speed_kph=50.0,
        game_display_size=(592, 444),
        observation_shape=(84, 84, 12),
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
        control_state=RaceControlState(),
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

    session_section = next(section for section in columns.left if section.title == "Run State")
    curriculum_line = next(line for line in session_section.lines if line.label == "Stage")

    assert curriculum_line.value == "lean_enabled"


def test_session_section_shows_checkpoint_experience_from_frames() -> None:
    columns = _build_panel_columns(
        episode=0,
        info={
            "frame_index": 0,
            "native_fps": 60.0,
        },
        reset_info={},
        episode_reward=0.0,
        paused=False,
        control_state=RaceControlState(),
        policy_curriculum_stage="lean_enabled",
        policy_num_timesteps=660_000,
        policy_experience_frames=1_320_000,
        policy_action=np.array([2, 1, 0], dtype=np.int64),
        policy_reload_age_seconds=5.0,
        policy_reload_error=None,
        action_repeat=2,
        stuck_min_speed_kph=50.0,
        game_display_size=(592, 444),
        observation_shape=(84, 116, 12),
        telemetry=_sample_telemetry(),
    )

    session_section = next(section for section in columns.left if section.title == "Run State")
    experience_line = next(line for line in session_section.lines if line.label == "Experience")

    assert experience_line.value == "6h 06m"


def test_session_section_keeps_mixed_action_repeat_lineage_experience() -> None:
    columns = _build_panel_columns(
        episode=0,
        info={
            "frame_index": 0,
            "native_fps": 60.0,
        },
        reset_info={},
        episode_reward=0.0,
        paused=False,
        control_state=RaceControlState(),
        policy_curriculum_stage="lean_enabled",
        policy_num_timesteps=10_000,
        policy_experience_frames=15_000,
        policy_action=np.array([2, 1, 0], dtype=np.int64),
        policy_reload_age_seconds=5.0,
        policy_reload_error=None,
        action_repeat=1,
        stuck_min_speed_kph=50.0,
        game_display_size=(592, 444),
        observation_shape=(84, 116, 12),
        telemetry=_sample_telemetry(),
    )

    session_section = next(section for section in columns.left if section.title == "Run State")
    experience_line = next(line for line in session_section.lines if line.label == "Experience")

    assert experience_line.value == "4m"


def test_session_section_shows_policy_deterministic_mode() -> None:
    columns = _build_panel_columns(
        episode=0,
        info={"frame_index": 0, "native_fps": 60.0},
        reset_info={},
        episode_reward=0.0,
        paused=False,
        control_state=RaceControlState(),
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

    session_section = next(section for section in columns.left if section.title == "Policy Output")
    deterministic_line = next(line for line in session_section.lines if line.label == "Mode")

    assert deterministic_line.value == "stochastic"


def test_session_section_shows_manual_driver_mode() -> None:
    columns = _build_panel_columns(
        episode=0,
        info={"frame_index": 0, "native_fps": 60.0},
        reset_info={},
        episode_reward=0.0,
        paused=False,
        control_state=RaceControlState(),
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

    session_section = next(section for section in columns.left if section.title == "Run State")
    driver_line = next(line for line in session_section.lines if line.label == "Driver")

    assert driver_line.value == "manual"


def test_session_section_formats_hybrid_action_value_with_fixed_digits() -> None:
    columns = _build_panel_columns(
        episode=0,
        info={"frame_index": 0, "native_fps": 60.0},
        reset_info={},
        episode_reward=0.0,
        paused=False,
        control_state=RaceControlState(),
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

    session_section = next(section for section in columns.left if section.title == "Policy Output")
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
        control_state=RaceControlState(),
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

    episode_section = next(
        section for section in columns.left if section.title == "Episode Progress"
    )
    step_line = next(line for line in episode_section.lines if line.label == "Step reward")
    return_line = next(line for line in episode_section.lines if line.label == "Return")

    assert step_line.value == "-0.0160"
    assert return_line.value == "-12.3457"


def test_session_section_omits_global_best_finish_position() -> None:
    columns = _build_panel_columns(
        episode=2,
        info={"frame_index": 0, "native_fps": 60.0},
        reset_info={},
        episode_reward=0.0,
        paused=False,
        control_state=RaceControlState(),
        policy_curriculum_stage=None,
        policy_action=np.array([2, 1, 0], dtype=np.int64),
        policy_reload_age_seconds=5.0,
        policy_reload_error=None,
        track_record_book=record_book(best_finish_position=8),
        action_repeat=1,
        stuck_min_speed_kph=50.0,
        game_display_size=(592, 444),
        observation_shape=(84, 116, 12),
        telemetry=_sample_telemetry(),
    )

    session_section = next(section for section in columns.left if section.title == "Run State")

    assert all(line.label != "Best position" for line in session_section.lines)


def test_format_reload_age_is_human_readable() -> None:
    assert _format_reload_age(None) == "manual"
    assert _format_reload_age(12.7) == "12s ago"
    assert _format_reload_age(125.0) == "2m 05s"
    assert _format_reload_age(3665.0) == "1h 01m"
