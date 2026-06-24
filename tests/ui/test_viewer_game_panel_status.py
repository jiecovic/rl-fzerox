# tests/ui/test_viewer_game_panel_status.py
from rl_fzerox.ui.watch.view.panels.core.model import _build_panel_columns
from tests.support.native_objects import encode_state_flags
from tests.ui.viewer_game_panel_support import (
    _career_section,
    _race_section,
    race_control_state,
)
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
        control_state=race_control_state(),
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
