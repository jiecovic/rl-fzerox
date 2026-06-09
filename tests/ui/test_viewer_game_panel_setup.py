# tests/ui/test_viewer_game_panel_setup.py
from rl_fzerox.ui.watch.view.panels.content.game import _format_telemetry_vehicle_setup
from rl_fzerox.ui.watch.view.panels.core.model import _build_panel_columns
from tests.support.native_objects import encode_state_flags
from tests.ui.viewer_game_panel_support import (
    _race_section,
    _setup_section,
    race_control_state,
)
from tests.ui.viewer_support import sample_telemetry as _sample_telemetry


def test_race_setup_uses_native_live_vehicle_setup_keys() -> None:
    columns = _build_panel_columns(
        episode=0,
        info={
            "frame_index": 0,
            "native_fps": 60.0,
            "racer_character_index": 0,
            "engine_setting_raw_value": 50,
        },
        reset_info={},
        episode_reward=0.0,
        paused=False,
        control_state=race_control_state(),
        policy_curriculum_stage=None,
        policy_action=None,
        policy_reload_age_seconds=None,
        policy_reload_error=None,
        action_repeat=3,
        stuck_min_speed_kph=50.0,
        game_display_size=(592, 444),
        observation_shape=(84, 116, 12),
        telemetry=_sample_telemetry(camera_setting_name="close_behind"),
    )

    values = {line.label: line.value for line in _setup_section(columns).lines}
    assert values["Vehicle"] == "Blue Falcon / Engine 50"
    assert values["Camera"] == "close behind"


def test_race_setup_prefers_live_telemetry_vehicle_setup() -> None:
    columns = _build_panel_columns(
        episode=0,
        info={
            "frame_index": 0,
            "native_fps": 60.0,
            "track_vehicle_name": "Blue Falcon",
            "racer_character_index_ram": 0,
            "engine_setting_percent_ram": 50.0,
        },
        reset_info={},
        episode_reward=0.0,
        paused=False,
        control_state=race_control_state(),
        policy_curriculum_stage=None,
        policy_action=None,
        policy_reload_age_seconds=None,
        policy_reload_error=None,
        action_repeat=3,
        stuck_min_speed_kph=50.0,
        game_display_size=(592, 444),
        observation_shape=(84, 116, 12),
        telemetry=_sample_telemetry(
            machine_character_index=3,
            engine_setting=1.0,
        ),
    )

    values = {line.label: line.value for line in _setup_section(columns).lines}
    assert values["Vehicle"] == "Fire Stingray / Engine 100"


def test_race_setup_ignores_missing_live_vehicle_setup_fields() -> None:
    class PlayerWithoutLiveSetup:
        pass

    class TelemetryWithoutLiveSetup:
        player = PlayerWithoutLiveSetup()

    assert _format_telemetry_vehicle_setup(TelemetryWithoutLiveSetup()) is None


def test_race_setup_does_not_fallback_to_configured_track_vehicle() -> None:
    columns = _build_panel_columns(
        episode=0,
        info={
            "frame_index": 0,
            "native_fps": 60.0,
            "track_vehicle_name": "Blue Falcon",
        },
        reset_info={},
        episode_reward=0.0,
        paused=False,
        control_state=race_control_state(),
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

    values = {line.label: line.value for line in _setup_section(columns).lines}
    assert values["Vehicle"] == "unknown"


def test_race_setup_camera_tracks_live_telemetry() -> None:
    common_args = {
        "episode": 0,
        "info": {"frame_index": 0, "native_fps": 60.0},
        "reset_info": {},
        "episode_reward": 0.0,
        "paused": False,
        "control_state": race_control_state(),
        "policy_curriculum_stage": None,
        "policy_action": None,
        "policy_reload_age_seconds": None,
        "policy_reload_error": None,
        "action_repeat": 3,
        "stuck_min_speed_kph": 50.0,
        "game_display_size": (592, 444),
        "observation_shape": (84, 116, 12),
    }
    first_columns = _build_panel_columns(
        **common_args,
        telemetry=_sample_telemetry(camera_setting_name="regular"),
    )
    second_columns = _build_panel_columns(
        **common_args,
        telemetry=_sample_telemetry(camera_setting_name="close_behind"),
    )

    first_values = {line.label: line.value for line in _setup_section(first_columns).lines}
    second_values = {line.label: line.value for line in _setup_section(second_columns).lines}
    assert first_values["Camera"] == "regular"
    assert second_values["Camera"] == "close behind"


def test_dirt_course_effect_lights_dirt_flag() -> None:
    columns = _build_panel_columns(
        episode=0,
        info={"frame_index": 0, "native_fps": 60.0},
        reset_info={},
        episode_reward=0.0,
        paused=False,
        control_state=race_control_state(),
        policy_curriculum_stage=None,
        policy_action=None,
        policy_reload_age_seconds=None,
        policy_reload_error=None,
        action_repeat=3,
        stuck_min_speed_kph=50.0,
        game_display_size=(592, 444),
        observation_shape=(84, 116, 12),
        telemetry=_sample_telemetry(
            state_flags=encode_state_flags(("active",)) | 2,
        ),
    )

    game_section = _race_section(columns)
    assert game_section.flag_viz is not None
    active_labels = {
        token.label for row in game_section.flag_viz.rows for token in row if token.active
    }
    assert "dirt" in active_labels
    assert "ice" not in active_labels


def test_ice_course_effect_lights_ice_flag() -> None:
    columns = _build_panel_columns(
        episode=0,
        info={"frame_index": 0, "native_fps": 60.0},
        reset_info={},
        episode_reward=0.0,
        paused=False,
        control_state=race_control_state(),
        policy_curriculum_stage=None,
        policy_action=None,
        policy_reload_age_seconds=None,
        policy_reload_error=None,
        action_repeat=3,
        stuck_min_speed_kph=50.0,
        game_display_size=(592, 444),
        observation_shape=(84, 116, 12),
        telemetry=_sample_telemetry(
            state_flags=encode_state_flags(("active",)) | 4,
        ),
    )

    game_section = _race_section(columns)
    assert game_section.flag_viz is not None
    active_labels = {
        token.label for row in game_section.flag_viz.rows for token in row if token.active
    }
    assert "ice" in active_labels
    assert "dirt" not in active_labels


def test_near_track_edge_lights_edge_flag() -> None:
    columns = _build_panel_columns(
        episode=0,
        info={"frame_index": 0, "native_fps": 60.0},
        reset_info={},
        episode_reward=0.0,
        paused=False,
        control_state=race_control_state(),
        policy_curriculum_stage=None,
        policy_action=None,
        policy_reload_age_seconds=None,
        policy_reload_error=None,
        action_repeat=3,
        stuck_min_speed_kph=50.0,
        game_display_size=(592, 444),
        observation_shape=(84, 116, 12),
        telemetry=_sample_telemetry(
            signed_lateral_offset=90.0,
            current_radius_left=100.0,
            current_radius_right=120.0,
        ),
    )

    game_section = _race_section(columns)
    assert game_section.flag_viz is not None
    active_labels = {
        token.label for row in game_section.flag_viz.rows for token in row if token.active
    }
    assert "edge" in active_labels
    assert "outside" not in active_labels


def test_outside_track_bounds_lights_outside_flag() -> None:
    columns = _build_panel_columns(
        episode=0,
        info={"frame_index": 0, "native_fps": 60.0},
        reset_info={},
        episode_reward=0.0,
        paused=False,
        control_state=race_control_state(),
        policy_curriculum_stage=None,
        policy_action=None,
        policy_reload_age_seconds=None,
        policy_reload_error=None,
        action_repeat=3,
        stuck_min_speed_kph=50.0,
        game_display_size=(592, 444),
        observation_shape=(84, 116, 12),
        telemetry=_sample_telemetry(
            signed_lateral_offset=-140.0,
            current_radius_left=100.0,
            current_radius_right=120.0,
        ),
    )

    game_section = _race_section(columns)
    assert game_section.flag_viz is not None
    active_labels = {
        token.label for row in game_section.flag_viz.rows for token in row if token.active
    }
    assert "edge" in active_labels
    assert "outside" in active_labels


def test_track_geometry_section_shows_racer_geometry() -> None:
    columns = _build_panel_columns(
        episode=0,
        info={"frame_index": 0, "native_fps": 60.0},
        reset_info={},
        episode_reward=0.0,
        paused=False,
        control_state=race_control_state(),
        policy_curriculum_stage=None,
        policy_action=None,
        policy_reload_age_seconds=None,
        policy_reload_error=None,
        action_repeat=3,
        stuck_min_speed_kph=50.0,
        game_display_size=(592, 444),
        observation_shape=(84, 116, 12),
        telemetry=_sample_telemetry(
            segment_index=8,
            segment_t=0.25,
            local_lateral_velocity=-9.5,
            signed_lateral_offset=90.0,
            lateral_distance=90.0,
            current_radius_left=100.0,
            current_radius_right=120.0,
            height_above_ground=6.0,
            velocity_magnitude=35.0,
            acceleration_magnitude=3.0,
            acceleration_force=1.5,
            drift_attack_force=0.25,
            damage_rumble_counter=1,
            recoil_tilt_magnitude=0.5,
            lap_distance=20_000.0,
            course_length=80_000.0,
            course_segment_count=64,
        ),
    )

    geometry_section = next(
        section for section in columns.middle if section.title == "Track Geometry"
    )
    values = {line.label: line.value for line in geometry_section.lines}
    assert values["Segment"] == "8 / 64"
    assert values["Spline t"] == "0.250"
    assert values["Lap progress"] == "25.0%"
    assert values["Center dist"] == "90.0"
    assert values["Lat vel"] == "-9.50"
    assert values["Sliding"] == "yes"
    assert values["Lat offset"] == "+90.0"
    assert values["Edge ratio"] == "left +0.90"
    assert values["Near edge"] == "left"
    assert values["Radius L/R"] == "100 / 120"
    assert values["Ground height"] == "6.0"
    assert values["Vel / acc"] == "35.00 / 3.00"
    assert values["Impact"] == "rumble 1 / recoil 0.500"


def test_game_section_shows_race_difficulty() -> None:
    columns = _build_panel_columns(
        episode=0,
        info={"frame_index": 0, "native_fps": 60.0},
        reset_info={},
        episode_reward=0.0,
        paused=False,
        control_state=race_control_state(),
        policy_curriculum_stage=None,
        policy_action=None,
        policy_reload_age_seconds=None,
        policy_reload_error=None,
        action_repeat=3,
        stuck_min_speed_kph=50.0,
        game_display_size=(592, 444),
        observation_shape=(84, 116, 12),
        telemetry=_sample_telemetry(difficulty_raw=2, difficulty_name="expert"),
    )

    game_section = _race_section(columns)
    difficulty_line = next(line for line in game_section.lines if line.label == "Difficulty")
    assert difficulty_line.value == "expert"


def test_game_section_shows_unknown_difficulty_raw_value() -> None:
    columns = _build_panel_columns(
        episode=0,
        info={"frame_index": 0, "native_fps": 60.0},
        reset_info={},
        episode_reward=0.0,
        paused=False,
        control_state=race_control_state(),
        policy_curriculum_stage=None,
        policy_action=None,
        policy_reload_age_seconds=None,
        policy_reload_error=None,
        action_repeat=3,
        stuck_min_speed_kph=50.0,
        game_display_size=(592, 444),
        observation_shape=(84, 116, 12),
        telemetry=_sample_telemetry(difficulty_raw=99, difficulty_name="unknown"),
    )

    game_section = _race_section(columns)
    difficulty_line = next(line for line in game_section.lines if line.label == "Difficulty")
    assert difficulty_line.value == "unknown (99)"


def test_game_section_hides_difficulty_for_time_attack() -> None:
    columns = _build_panel_columns(
        episode=0,
        info={"frame_index": 0, "native_fps": 60.0},
        reset_info={},
        episode_reward=0.0,
        paused=False,
        control_state=race_control_state(),
        policy_curriculum_stage=None,
        policy_action=None,
        policy_reload_age_seconds=None,
        policy_reload_error=None,
        action_repeat=3,
        stuck_min_speed_kph=50.0,
        game_display_size=(592, 444),
        observation_shape=(84, 116, 12),
        telemetry=_sample_telemetry(
            game_mode_raw=14,
            game_mode_name="time_attack",
            difficulty_raw=0,
            difficulty_name="novice",
        ),
    )

    game_section = _race_section(columns)
    assert all(line.label != "Difficulty" for line in game_section.lines)


def test_game_section_shows_camera_setting() -> None:
    columns = _build_panel_columns(
        episode=0,
        info={"frame_index": 0, "native_fps": 60.0},
        reset_info={},
        episode_reward=0.0,
        paused=False,
        control_state=race_control_state(),
        policy_curriculum_stage=None,
        policy_action=None,
        policy_reload_age_seconds=None,
        policy_reload_error=None,
        action_repeat=3,
        stuck_min_speed_kph=50.0,
        game_display_size=(592, 444),
        observation_shape=(84, 116, 12),
        telemetry=_sample_telemetry(camera_setting_raw=1, camera_setting_name="close_behind"),
    )

    setup_section = _setup_section(columns)
    camera_line = next(line for line in setup_section.lines if line.label == "Camera")
    assert camera_line.value == "close behind"


def test_game_section_prefers_live_vehicle_setup_over_track_metadata() -> None:
    columns = _build_panel_columns(
        episode=0,
        info={
            "frame_index": 0,
            "native_fps": 60.0,
            "track_vehicle_name": "Blue Falcon",
            "racer_character_index_ram": 3,
            "engine_setting_percent_ram": 100.0,
        },
        reset_info={},
        episode_reward=0.0,
        paused=False,
        control_state=race_control_state(),
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

    setup_section = _setup_section(columns)
    vehicle_line = next(line for line in setup_section.lines if line.label == "Vehicle")
    assert vehicle_line.value == "Fire Stingray / Engine 100"


def test_game_section_falls_back_to_menu_vehicle_when_live_racer_is_uninitialized() -> None:
    columns = _build_panel_columns(
        episode=0,
        info={
            "frame_index": 0,
            "native_fps": 60.0,
            "track_vehicle_name": "Blue Falcon",
            "racer_character_index_ram": -1,
            "player_character_index_ram": 7,
            "engine_setting_percent_ram": 75.0,
        },
        reset_info={},
        episode_reward=0.0,
        paused=False,
        control_state=race_control_state(),
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

    setup_section = _setup_section(columns)
    vehicle_line = next(line for line in setup_section.lines if line.label == "Vehicle")
    assert vehicle_line.value == "Iron Tiger / Engine 75"


def test_game_section_ignores_invalid_live_engine_value() -> None:
    columns = _build_panel_columns(
        episode=0,
        info={
            "frame_index": 0,
            "native_fps": 60.0,
            "track_vehicle_name": "Blue Falcon",
            "racer_character_index_ram": -1,
            "player_character_index_ram": 0,
            "engine_setting_percent_ram": 5000.0,
        },
        reset_info={},
        episode_reward=0.0,
        paused=False,
        control_state=race_control_state(),
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

    setup_section = _setup_section(columns)
    vehicle_line = next(line for line in setup_section.lines if line.label == "Vehicle")
    assert vehicle_line.value == "Blue Falcon"


def test_game_section_shows_unknown_camera_setting_raw_value() -> None:
    columns = _build_panel_columns(
        episode=0,
        info={"frame_index": 0, "native_fps": 60.0},
        reset_info={},
        episode_reward=0.0,
        paused=False,
        control_state=race_control_state(),
        policy_curriculum_stage=None,
        policy_action=None,
        policy_reload_age_seconds=None,
        policy_reload_error=None,
        action_repeat=3,
        stuck_min_speed_kph=50.0,
        game_display_size=(592, 444),
        observation_shape=(84, 116, 12),
        telemetry=_sample_telemetry(camera_setting_raw=99, camera_setting_name="unknown"),
    )

    setup_section = _setup_section(columns)
    camera_line = next(line for line in setup_section.lines if line.label == "Camera")
    assert camera_line.value == "unknown (99)"


def test_game_section_shows_position_out_of_total_racers() -> None:
    columns = _build_panel_columns(
        episode=0,
        info={"frame_index": 0, "native_fps": 60.0},
        reset_info={},
        episode_reward=0.0,
        paused=False,
        control_state=race_control_state(),
        policy_curriculum_stage=None,
        policy_action=None,
        policy_reload_age_seconds=None,
        policy_reload_error=None,
        action_repeat=3,
        stuck_min_speed_kph=50.0,
        game_display_size=(592, 444),
        observation_shape=(84, 116, 12),
        telemetry=_sample_telemetry(position=1, total_racers=30),
    )

    game_section = _race_section(columns)
    position_line = next(line for line in game_section.lines if line.label == "Position")
    assert position_line.value == "1 / 30"
