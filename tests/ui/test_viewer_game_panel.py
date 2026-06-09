# tests/ui/test_viewer_game_panel.py
from fzerox_emulator import RaceControlState
from rl_fzerox.core.envs.actions import RACE_CONTROL_MASKS
from rl_fzerox.ui.watch.records import track_record_key
from rl_fzerox.ui.watch.view.panels.content.game import _format_telemetry_vehicle_setup
from rl_fzerox.ui.watch.view.panels.core.model import _build_panel_columns
from rl_fzerox.ui.watch.view.panels.rendering.draw import _record_tab_sections
from rl_fzerox.ui.watch.view.panels.visuals.viz import _control_viz
from rl_fzerox.ui.watch.view.screen.theme import PALETTE
from rl_fzerox.ui.watch.view.screen.types import PanelColumns, PanelSection
from tests.support.native_objects import encode_state_flags
from tests.ui.viewer_support import record_book, record_entry
from tests.ui.viewer_support import sample_telemetry as _sample_telemetry


def race_control_state(
    *,
    control_mask: int = 0,
    stick_x: float = 0.0,
    pitch: float = 0.0,
) -> RaceControlState:
    return RaceControlState.from_mask(
        control_mask,
        stick_x=stick_x,
        pitch=pitch,
    )


def _race_section(columns: PanelColumns) -> PanelSection:
    return next(section for section in columns.left if section.title == "Race State")


def _setup_section(columns: PanelColumns) -> PanelSection:
    return next(section for section in columns.left if section.title == "Race Setup")


def _career_section(columns: PanelColumns, title: str) -> PanelSection:
    return next(section for section in columns.career if section.title == title)


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
            state_labels=("active", "dash_pad_boost", "collision_recoil"),
            reverse_timer=40,
        ),
    )

    game_section = _race_section(columns)
    assert game_section.flag_viz is not None
    assert len(game_section.flag_viz.rows) == 6
    active_labels = {
        token.label for row in game_section.flag_viz.rows for token in row if token.active
    }
    assert {"boost", "energy loss", "damage taken", "recoil", "reverse", "slow"}.issubset(
        active_labels
    )
    assert "dash" not in active_labels


def test_career_mode_panel_shows_structured_fsm_facts() -> None:
    columns = _build_panel_columns(
        episode=0,
        info={
            "career_mode_attempt_id": "attempt-a",
            "career_mode_completed_targets": 1,
            "career_mode_fsm_awaiting_fresh_race": True,
            "career_mode_fsm_camera_synced": False,
            "career_mode_fsm_camera_target": "close_behind",
            "career_mode_fsm_completed_laps": 3,
            "career_mode_fsm_completion_fraction": 1.0,
            "career_mode_fsm_continuing_result": True,
            "career_mode_fsm_course_index": 0,
            "career_mode_fsm_fresh_race_ready": False,
            "career_mode_fsm_game_mode": "gp_race",
            "career_mode_fsm_intro_timer": 0,
            "career_mode_fsm_pending_steps": 1,
            "career_mode_fsm_race_time_ms": 94_200.0,
            "career_mode_fsm_selected_mode_raw": 0,
            "career_mode_fsm_terminal_reason": "finished",
            "career_mode_fsm_terminal_result": True,
            "career_mode_fsm_total_laps": 3,
            "career_mode_fsm_transition_raw": 0,
            "career_mode_inspection_status": "partial",
            "career_mode_next_target_label": "Clear Novice Queen Cup",
            "career_mode_phase": "continue_after_race",
            "career_mode_policy_active": False,
            "career_mode_target_label": "Clear Novice Jack Cup",
            "career_mode_total_targets": 16,
            "frame_index": 0,
            "native_fps": 60.0,
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

    controller_values = {
        line.label: line.value for line in _career_section(columns, "Career Controller").lines
    }
    policy_values = {
        line.label: line.value for line in _career_section(columns, "Career Policy").lines
    }

    assert "mode=gp_race" in controller_values["Game facts"]
    assert "terminal=yes" in controller_values["Race boundary"]
    assert "fresh=no" in controller_values["Race boundary"]
    assert "laps=3 / 3" in controller_values["Race progress"]
    assert "comp=100.0%" in controller_values["Race progress"]
    assert "target=close_behind" in controller_values["Camera"]
    assert policy_values["Policy control"] == "inactive"


def test_energy_refill_course_effect_lights_refill_flag() -> None:
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
            state_flags=encode_state_flags(("active",)) | 1,
            energy=150.0,
            max_energy=178.0,
        ),
    )

    game_section = _race_section(columns)
    assert game_section.flag_viz is not None
    active_labels = {
        token.label for row in game_section.flag_viz.rows for token in row if token.active
    }
    assert "refill" in active_labels


def test_refill_surface_flag_ignores_energy_fullness() -> None:
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
            state_flags=encode_state_flags(("active",)) | 1,
            energy=178.0,
            max_energy=178.0,
        ),
    )

    game_section = _race_section(columns)
    assert game_section.flag_viz is not None
    active_labels = {
        token.label for row in game_section.flag_viz.rows for token in row if token.active
    }
    assert "refill" in active_labels


def test_game_section_shows_current_and_max_progress() -> None:
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
            race_distance=100_000.0,
            lap_distance=20_000.0,
            course_length=80_000.0,
            total_lap_count=3,
        ),
    )

    values = {line.label: line.value for line in _race_section(columns).lines}
    assert values["Total progress"] == "100,000.0 / 240,000.0"
    assert values["Lap progress"] == "20,000.0 / 80,000.0"


def test_game_section_shows_ko_star_ram_count() -> None:
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
        telemetry=_sample_telemetry(ko_star_count=4),
    )

    values = {line.label: line.value for line in _race_section(columns).lines}
    assert values["KO stars"] == "4"


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
        track_pool_records=(
            {
                "track_display_name": "Big Blue Time Attack - Blue Falcon Engine 50",
                "track_id": "big_blue",
                "track_non_agg_best_time_ms": 48035,
                "track_non_agg_worst_time_ms": 75846,
            },
        ),
    )

    records_section = next(section for section in columns.records if section.title == "Records")
    header_line = next(line for line in records_section.lines if line.label == "Big Blue")
    pb_line = next(line for line in records_section.lines if line.label == "Best time")
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
        track_record_book=record_book(
            {
                "silence": record_entry(
                    best_finish_time_ms=98765,
                    latest_finish_time_ms=101234,
                    attempt_stats={
                        "attempts": 3,
                        "finishes": 1,
                        "completion_samples": 3,
                        "completion_sum": 2.25,
                        "best_completion": 1.0,
                    },
                )
            }
        ),
        track_pool_records=(
            {
                "track_id": "silence",
                "track_display_name": "Silence Time Attack - Blue Falcon Engine 50",
                "track_non_agg_best_time_ms": 60638,
                "track_non_agg_worst_time_ms": 63279,
            },
        ),
    )

    records_section = next(section for section in columns.records if section.title == "Records")
    header_line = next(line for line in records_section.lines if line.label == "> Silence")
    pb_line = next(line for line in records_section.lines if line.label == "Best time")
    latest_line = next(line for line in records_section.lines if line.label == "Latest")
    attempts_line = next(line for line in records_section.lines if line.label == "Attempts")
    assert header_line.value == ""
    assert header_line.status_icon == "outside"
    assert header_line.status_text == "+35.5s"
    assert pb_line.value == "1:38.765"
    assert latest_line.value == "1:41.234 (+2.5s)"
    assert attempts_line.value == "3 · finish 33.3% · comp 75.0%"


def test_records_section_shows_gp_best_rank_with_watch_best_time() -> None:
    columns = _build_panel_columns(
        episode=0,
        info={
            "frame_index": 0,
            "native_fps": 60.0,
            "track_id": "silence",
            "track_mode": "gp_race",
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
        track_record_book=record_book(
            {
                "silence": record_entry(
                    best_finish_rank=1,
                    best_finish_rank_time_ms=101_000,
                    best_finish_rank_setup={
                        "vehicle_name": "Deep Claw",
                        "engine_setting_raw_value": 60,
                    },
                    best_finish_time_ms=98765,
                    best_finish_time_rank=2,
                    best_finish_time_setup={
                        "vehicle_name": "Blue Falcon",
                        "engine_setting_raw_value": 50,
                    },
                )
            }
        ),
        track_pool_records=(
            {
                "track_id": "silence",
                "track_display_name": "Silence GP Race - Blue Falcon Engine 50",
                "track_mode": "gp_race",
            },
        ),
    )

    records_section = next(section for section in columns.records if section.title == "Records")
    best_time_line = next(line for line in records_section.lines if line.label == "Best time")
    best_position_line = next(line for line in records_section.lines if line.label == "Best pos")

    assert best_time_line.value == "1:38.765 · P2 · Blue Falcon / Engine 50"
    assert best_position_line.value == "P1 · 1:41.000 · Deep Claw / Engine 60"


def test_records_section_groups_track_pool_by_cup() -> None:
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
        telemetry=_sample_telemetry(),
        track_pool_records=(
            {
                "track_id": "port_town_2",
                "track_course_ref": "queen/port_town_2",
                "track_course_name": "Port Town II",
            },
            {
                "track_id": "mute_city",
                "track_course_ref": "jack/mute_city",
                "track_course_name": "Mute City",
            },
            {
                "track_id": "silence",
                "track_course_ref": "jack/silence",
                "track_course_name": "Silence",
            },
        ),
    )

    assert [section.title for section in columns.records] == ["Jack Cup", "Queen Cup"]
    assert [line.label for line in columns.records[0].lines if line.heading] == [
        "Mute City",
        "Silence",
    ]
    assert [line.label for line in columns.records[1].lines if line.heading] == ["Port Town II"]
    assert [section.title for section in _record_tab_sections(columns.records, 1)] == ["Queen Cup"]


def test_records_section_groups_generated_x_cup_records() -> None:
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
        telemetry=_sample_telemetry(),
        track_pool_records=(
            {
                "track_id": "mute_city",
                "track_course_ref": "jack/mute_city",
                "track_course_name": "Mute City",
            },
            {
                "track_id": "x_cup_d6a1a626",
                "track_course_id": "x_cup_d6a1a626",
                "track_runtime_course_key": "x_cup_slot_1",
                "track_course_name": "Space Plant",
                "track_course_index": 48,
                "track_generated_course_kind": "x_cup",
            },
        ),
    )

    assert [section.title for section in columns.records] == ["Jack Cup", "X Cup"]
    assert [line.label for line in columns.records[1].lines if line.heading] == ["Space Plant"]


def test_records_section_dedupes_course_variants_to_one_course_row() -> None:
    columns = _build_panel_columns(
        episode=0,
        info={"frame_index": 0, "native_fps": 60.0, "track_course_id": "mute_city"},
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
        track_record_book=record_book({"mute_city": record_entry(best_finish_time_ms=88_000)}),
        track_pool_records=(
            {
                "track_id": "mute_city_blue_falcon",
                "track_course_id": "mute_city",
                "track_course_name": "Mute City",
                "track_vehicle": "blue_falcon",
            },
            {
                "track_id": "mute_city_fire_stingray",
                "track_course_id": "mute_city",
                "track_course_name": "Mute City",
                "track_vehicle": "fire_stingray",
            },
        ),
    )

    records_section = next(section for section in columns.records if section.title == "Records")
    headings = [line.label for line in records_section.lines if line.heading]

    assert headings == ["> Mute City"]


def test_records_section_follows_selected_gp_difficulty() -> None:
    novice_record: dict[str, object] = {
        "track_id": "mute_city_novice",
        "track_course_id": "mute_city",
        "track_course_name": "Mute City",
        "track_mode": "gp_race",
        "track_gp_difficulty": "novice",
    }
    expert_record: dict[str, object] = {
        "track_id": "mute_city_expert",
        "track_course_id": "mute_city",
        "track_course_name": "Mute City",
        "track_mode": "gp_race",
        "track_gp_difficulty": "expert",
    }
    novice_key = track_record_key(novice_record)
    expert_key = track_record_key(expert_record)
    assert novice_key is not None
    assert expert_key is not None
    columns = _build_panel_columns(
        episode=0,
        info={
            "frame_index": 0,
            "native_fps": 60.0,
            "track_course_id": "mute_city",
            "track_mode": "gp_race",
            "track_gp_difficulty": "expert",
            "watch_selected_gp_difficulty": "expert",
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
        telemetry=_sample_telemetry(difficulty_name="expert", difficulty_raw=2),
        track_record_book=record_book(
            {
                novice_key: record_entry(
                    best_finish_rank=3,
                    best_finish_rank_time_ms=92_000,
                    best_finish_time_ms=92_000,
                    best_finish_time_rank=3,
                ),
                expert_key: record_entry(
                    best_finish_rank=1,
                    best_finish_rank_time_ms=102_000,
                    best_finish_time_ms=98_000,
                    best_finish_time_rank=1,
                ),
            }
        ),
        track_pool_records=(novice_record, expert_record),
    )

    assert [section.title for section in columns.records] == ["Records"]
    headings = [line.label for line in columns.records[0].lines if line.heading]
    expert_pb_line = next(line for line in columns.records[0].lines if line.label == "Best time")
    expert_pos_line = next(line for line in columns.records[0].lines if line.label == "Best pos")

    assert headings == ["> Mute City"]
    assert expert_pb_line.value == "1:38.000 · P1"
    assert expert_pos_line.value == "P1 · 1:42.000"


def test_records_section_highlights_current_track_heading() -> None:
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
        track_pool_records=(
            {
                "track_id": "mute_city",
                "track_course_name": "Mute City",
            },
            {
                "track_id": "silence",
                "track_course_name": "Silence",
            },
        ),
    )

    records_section = next(section for section in columns.records if section.title == "Records")
    mute_line = next(line for line in records_section.lines if line.label == "Mute City")
    silence_line = next(line for line in records_section.lines if line.label == "> Silence")

    assert mute_line.label_color is None
    assert silence_line.label_color == PALETTE.text_accent
    assert silence_line.status_text == "LIVE"


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
        track_record_book=record_book(
            {
                "silence": record_entry(
                    best_finish_time_ms=97_530,
                    latest_finish_time_ms=97_530,
                    latest_finish_delta_ms=-1_235,
                )
            }
        ),
        track_pool_records=(
            {
                "track_id": "silence",
                "track_display_name": "Silence Time Attack - Blue Falcon Engine 50",
                "track_non_agg_best_time_ms": 60638,
                "track_non_agg_worst_time_ms": 63279,
            },
        ),
    )

    records_section = next(section for section in columns.records if section.title == "Records")
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
        track_record_book=record_book({"silence": record_entry(best_finish_time_ms=62_000)}),
        track_pool_records=(
            {
                "track_id": "silence",
                "track_display_name": "Silence Time Attack - Blue Falcon Engine 50",
                "track_non_agg_best_time_ms": 60638,
                "track_non_agg_worst_time_ms": 63279,
            },
        ),
    )

    records_section = next(section for section in columns.records if section.title == "Records")
    header_line = next(line for line in records_section.lines if line.label == "> Silence")
    pb_line = next(line for line in records_section.lines if line.label == "Best time")
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
        track_record_book=record_book({"silence": record_entry(best_finish_time_ms=138_379)}),
        track_pool_records=(
            {
                "track_id": "silence",
                "track_display_name": "Silence Time Attack - Blue Falcon Engine 50",
                "track_non_agg_best_time_ms": 60638,
                "track_non_agg_worst_time_ms": 63279,
            },
        ),
    )

    records_section = next(section for section in columns.records if section.title == "Records")
    header_line = next(line for line in records_section.lines if line.label == "> Silence")
    assert header_line.status_icon == "outside"
    assert header_line.status_text == "+1min 15.1s"


def test_dash_pad_boost_lights_single_boost_pill() -> None:
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
        telemetry=_sample_telemetry(state_labels=("active", "dash_pad_boost"), boost_timer=12),
    )

    game_section = _race_section(columns)
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
        control_state=race_control_state(),
        policy_curriculum_stage=None,
        policy_action=None,
        policy_reload_age_seconds=None,
        policy_reload_error=None,
        action_repeat=3,
        stuck_min_speed_kph=50.0,
        game_display_size=(592, 444),
        observation_shape=(84, 116, 12),
        telemetry=_sample_telemetry(boost_timer=12),
    )

    game_section = _race_section(columns)
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
        control_state=race_control_state(),
        policy_curriculum_stage=None,
        policy_action=None,
        policy_reload_age_seconds=None,
        policy_reload_error=None,
        action_repeat=3,
        stuck_min_speed_kph=50.0,
        game_display_size=(592, 444),
        observation_shape=(84, 116, 12),
        telemetry=_sample_telemetry(boost_timer=-1),
    )

    game_section = _race_section(columns)
    assert game_section.flag_viz is not None
    active_labels = {
        token.label for row in game_section.flag_viz.rows for token in row if token.active
    }
    assert "boost" in active_labels


def test_control_viz_snaps_discrete_gas_to_button_state() -> None:
    viz = _control_viz(
        race_control_state(control_mask=RACE_CONTROL_MASKS.accelerate),
        gas_level=0.37,
        continuous_drive_enabled=False,
        force_full_throttle=False,
    )

    assert viz.gas_level == 1.0
