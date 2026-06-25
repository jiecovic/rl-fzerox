# tests/ui/test_viewer_state_vector.py
"""Watch viewer tests for state-vector panel grouping and toggles.

These cases cover component grouping, action-history separation, zeroed feature
markers, and clickable Watch ablation toggles.
"""

import numpy as np

from fzerox_emulator import RaceControlState
from rl_fzerox.ui.watch.view.panels.content.state_vector import policy_state_sections
from rl_fzerox.ui.watch.view.panels.core.model import _build_panel_columns
from tests.ui.viewer_support import panel_group_labels as _panel_group_labels
from tests.ui.viewer_support import panel_group_values as _panel_group_values
from tests.ui.viewer_support import sample_telemetry as _sample_telemetry


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


def test_side_panel_can_show_policy_observation_state_vector() -> None:
    feature_names = (
        "vehicle_state.speed_norm",
        "vehicle_state.energy_frac",
        "vehicle_state.reverse_active",
        "vehicle_state.airborne",
        "vehicle_state.can_boost",
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
        "speed_norm": "0.5000",
        "energy_frac": "0.7500",
        "reverse_active": "1.0000",
        "airborne": "0.0000",
        "can_boost": "1.0000",
        "boost_active": "0.0000",
        "lateral_velocity_norm": "0.2500",
        "sliding_active": "1.0000",
    }


def test_side_panel_splits_component_action_history_from_state_vector() -> None:
    feature_names = (
        "vehicle_state.speed_norm",
        "vehicle_state.energy_frac",
        "vehicle_state.reverse_active",
        "vehicle_state.airborne",
        "vehicle_state.can_boost",
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
        "can_boost",
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
