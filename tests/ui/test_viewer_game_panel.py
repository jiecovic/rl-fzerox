# tests/ui/test_viewer_game_panel.py
from fzerox_emulator import ControllerState
from rl_fzerox.ui.watch.view.panels.model import _build_panel_columns
from tests.support.native_objects import encode_state_flags
from tests.ui.viewer_support import sample_telemetry as _sample_telemetry


def test_game_flags_are_rendered_in_fixed_rows() -> None:
    columns = _build_panel_columns(
        episode=0,
        info={
            "frame_index": 0,
            "native_fps": 60.0,
            "energy_loss_total": 0.25,
            "damage_taken_frames": 1,
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
        stuck_step_limit=240,
        stuck_min_speed_kph=50.0,
        game_display_size=(592, 444),
        observation_shape=(84, 116, 12),
        telemetry=_sample_telemetry(
            state_labels=("active", "dash_pad_boost", "collision_recoil"),
            reverse_timer=40,
        ),
    )

    game_section = columns.middle[0]
    assert game_section.flag_viz is not None
    assert len(game_section.flag_viz.rows) == 6
    active_labels = {
        token.label for row in game_section.flag_viz.rows for token in row if token.active
    }
    assert {"boost", "energy loss", "damage taken", "recoil", "reverse", "slow"}.issubset(
        active_labels
    )
    assert "dash" not in active_labels


def test_energy_refill_course_effect_lights_refill_flag() -> None:
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
        telemetry=_sample_telemetry(
            state_flags=encode_state_flags(("active",)) | 1,
            energy=150.0,
            max_energy=178.0,
        ),
    )

    game_section = columns.middle[0]
    assert game_section.flag_viz is not None
    active_labels = {
        token.label for row in game_section.flag_viz.rows for token in row if token.active
    }
    assert "refill" in active_labels


def test_energy_refill_course_effect_stays_off_when_energy_is_full() -> None:
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
        telemetry=_sample_telemetry(
            state_flags=encode_state_flags(("active",)) | 1,
            energy=178.0,
            max_energy=178.0,
        ),
    )

    game_section = columns.middle[0]
    assert game_section.flag_viz is not None
    active_labels = {
        token.label for row in game_section.flag_viz.rows for token in row if token.active
    }
    assert "refill" not in active_labels


def test_dirt_course_effect_lights_dirt_flag() -> None:
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
        telemetry=_sample_telemetry(
            state_flags=encode_state_flags(("active",)) | 2,
        ),
    )

    game_section = columns.middle[0]
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
        telemetry=_sample_telemetry(
            state_flags=encode_state_flags(("active",)) | 4,
        ),
    )

    game_section = columns.middle[0]
    assert game_section.flag_viz is not None
    active_labels = {
        token.label for row in game_section.flag_viz.rows for token in row if token.active
    }
    assert "ice" in active_labels
    assert "dirt" not in active_labels


def test_track_geometry_section_shows_racer_geometry() -> None:
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
        ),
    )

    geometry_section = next(
        section for section in columns.middle if section.title == "Track Geometry"
    )
    values = {line.label: line.value for line in geometry_section.lines}
    assert values["Segment"] == "8"
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
        telemetry=_sample_telemetry(difficulty_raw=2, difficulty_name="expert"),
    )

    game_section = columns.middle[0]
    difficulty_line = next(line for line in game_section.lines if line.label == "Difficulty")
    assert difficulty_line.value == "expert"


def test_game_section_shows_unknown_difficulty_raw_value() -> None:
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
        telemetry=_sample_telemetry(difficulty_raw=99, difficulty_name="unknown"),
    )

    game_section = columns.middle[0]
    difficulty_line = next(line for line in game_section.lines if line.label == "Difficulty")
    assert difficulty_line.value == "unknown (99)"


def test_game_section_shows_camera_setting() -> None:
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
        telemetry=_sample_telemetry(camera_setting_raw=1, camera_setting_name="close_behind"),
    )

    game_section = columns.middle[0]
    camera_line = next(line for line in game_section.lines if line.label == "Camera")
    assert camera_line.value == "close behind"


def test_game_section_shows_unknown_camera_setting_raw_value() -> None:
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
        telemetry=_sample_telemetry(camera_setting_raw=99, camera_setting_name="unknown"),
    )

    game_section = columns.middle[0]
    camera_line = next(line for line in game_section.lines if line.label == "Camera")
    assert camera_line.value == "unknown (99)"


def test_game_section_shows_position_out_of_total_racers() -> None:
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
        telemetry=_sample_telemetry(position=1, total_racers=30),
    )

    game_section = columns.middle[0]
    position_line = next(line for line in game_section.lines if line.label == "Position")
    assert position_line.value == "1 / 30"


def test_records_section_shows_non_agg_reference_records() -> None:
    columns = _build_panel_columns(
        episode=0,
        info={
            "frame_index": 0,
            "native_fps": 60.0,
            "track_non_agg_best_time_ms": 48035,
            "track_non_agg_best_player": "Daniel",
            "track_non_agg_worst_time_ms": 75846,
            "track_non_agg_worst_player": "FTQ",
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
        stuck_step_limit=240,
        stuck_min_speed_kph=50.0,
        game_display_size=(592, 444),
        observation_shape=(84, 116, 12),
        telemetry=_sample_telemetry(),
        track_pool_records=(
            {
                "track_display_name": "Big Blue Time Attack - Blue Falcon Balanced",
                "track_id": "big_blue",
                "track_non_agg_best_time_ms": 48035,
                "track_non_agg_worst_time_ms": 75846,
            },
        ),
    )

    records_section = next(section for section in columns.left if section.title == "Records")
    header_line = next(line for line in records_section.lines if line.label == "Big Blue")
    pb_line = next(line for line in records_section.lines if line.label == "PB")
    wr_line = next(line for line in records_section.lines if line.label == "WR")
    assert header_line.value == ""
    assert header_line.status_icon == "none"
    assert header_line.status_text == ""
    assert pb_line.value == "--"
    assert wr_line.value == "48.035 - 1:15.846"


def test_records_section_shows_watch_best_for_track_pool() -> None:
    columns = _build_panel_columns(
        episode=0,
        info={
            "frame_index": 0,
            "native_fps": 60.0,
            "track_id": "silence",
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
        stuck_step_limit=240,
        stuck_min_speed_kph=50.0,
        game_display_size=(592, 444),
        observation_shape=(84, 116, 12),
        telemetry=_sample_telemetry(),
        best_finish_times={"silence": 98765},
        latest_finish_times={"silence": 101234},
        track_pool_records=(
            {
                "track_id": "silence",
                "track_display_name": "Silence Time Attack - Blue Falcon Balanced",
                "track_non_agg_best_time_ms": 60638,
                "track_non_agg_worst_time_ms": 63279,
            },
        ),
    )

    records_section = next(section for section in columns.left if section.title == "Records")
    header_line = next(line for line in records_section.lines if line.label == "Silence")
    pb_line = next(line for line in records_section.lines if line.label == "PB")
    latest_line = next(line for line in records_section.lines if line.label == "Latest")
    assert header_line.value == ""
    assert header_line.status_icon == "outside"
    assert header_line.status_text == "+35.5s"
    assert pb_line.value == "1:38.765"
    assert latest_line.value == "1:41.234 (+2.5s)"


def test_records_section_shows_latest_improvement_against_previous_pb() -> None:
    columns = _build_panel_columns(
        episode=0,
        info={
            "frame_index": 0,
            "native_fps": 60.0,
            "track_id": "silence",
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
        stuck_step_limit=240,
        stuck_min_speed_kph=50.0,
        game_display_size=(592, 444),
        observation_shape=(84, 116, 12),
        telemetry=_sample_telemetry(),
        best_finish_times={"silence": 97_530},
        latest_finish_times={"silence": 97_530},
        latest_finish_deltas_ms={"silence": -1_235},
        track_pool_records=(
            {
                "track_id": "silence",
                "track_display_name": "Silence Time Attack - Blue Falcon Balanced",
                "track_non_agg_best_time_ms": 60638,
                "track_non_agg_worst_time_ms": 63279,
            },
        ),
    )

    records_section = next(section for section in columns.left if section.title == "Records")
    latest_line = next(line for line in records_section.lines if line.label == "Latest")
    assert latest_line.value == "1:37.530 (-1.2s)"


def test_records_section_marks_watch_best_inside_reference_range() -> None:
    columns = _build_panel_columns(
        episode=0,
        info={
            "frame_index": 0,
            "native_fps": 60.0,
            "track_id": "silence",
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
        stuck_step_limit=240,
        stuck_min_speed_kph=50.0,
        game_display_size=(592, 444),
        observation_shape=(84, 116, 12),
        telemetry=_sample_telemetry(),
        best_finish_times={"silence": 62_000},
        track_pool_records=(
            {
                "track_id": "silence",
                "track_display_name": "Silence Time Attack - Blue Falcon Balanced",
                "track_non_agg_best_time_ms": 60638,
                "track_non_agg_worst_time_ms": 63279,
            },
        ),
    )

    records_section = next(section for section in columns.left if section.title == "Records")
    header_line = next(line for line in records_section.lines if line.label == "Silence")
    pb_line = next(line for line in records_section.lines if line.label == "PB")
    assert header_line.value == ""
    assert header_line.status_icon == "in_range"
    assert header_line.status_text == "+1.4s"
    assert pb_line.value == "1:02.000"


def test_records_section_formats_minute_scale_reference_gap() -> None:
    columns = _build_panel_columns(
        episode=0,
        info={
            "frame_index": 0,
            "native_fps": 60.0,
            "track_id": "silence",
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
        stuck_step_limit=240,
        stuck_min_speed_kph=50.0,
        game_display_size=(592, 444),
        observation_shape=(84, 116, 12),
        telemetry=_sample_telemetry(),
        best_finish_times={"silence": 138_379},
        track_pool_records=(
            {
                "track_id": "silence",
                "track_display_name": "Silence Time Attack - Blue Falcon Balanced",
                "track_non_agg_best_time_ms": 60638,
                "track_non_agg_worst_time_ms": 63279,
            },
        ),
    )

    records_section = next(section for section in columns.left if section.title == "Records")
    header_line = next(line for line in records_section.lines if line.label == "Silence")
    assert header_line.status_icon == "outside"
    assert header_line.status_text == "+1min 15.1s"


def test_dash_pad_boost_lights_single_boost_pill() -> None:
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
        telemetry=_sample_telemetry(state_labels=("active", "dash_pad_boost"), boost_timer=12),
    )

    game_section = columns.middle[0]
    assert game_section.flag_viz is not None
    active_labels = {
        token.label for row in game_section.flag_viz.rows for token in row if token.active
    }
    assert "boost" in active_labels
    assert "dash" not in active_labels


def test_manual_boost_timer_lights_boost_pill() -> None:
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
        telemetry=_sample_telemetry(boost_timer=12),
    )

    game_section = columns.middle[0]
    assert game_section.flag_viz is not None
    active_labels = {
        token.label for row in game_section.flag_viz.rows for token in row if token.active
    }
    assert "boost" in active_labels


def test_signed_boost_timer_lights_generic_boost_pill() -> None:
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
        telemetry=_sample_telemetry(boost_timer=-1),
    )

    game_section = columns.middle[0]
    assert game_section.flag_viz is not None
    active_labels = {
        token.label for row in game_section.flag_viz.rows for token in row if token.active
    }
    assert "boost" in active_labels
