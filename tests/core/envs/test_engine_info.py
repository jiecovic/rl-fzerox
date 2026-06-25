# tests/core/envs/test_engine_info.py
from __future__ import annotations

import pytest

from rl_fzerox.core.envs.engine.info import backend_step_info, telemetry_info
from tests.support.native_objects import make_telemetry


class _Backend:
    name = "fake"
    frame_index = 12
    display_aspect_ratio = 4 / 3
    native_fps = 60.0

    def vehicle_setup_info(self) -> dict[str, object]:
        return {
            "player_character_index_ram": 0,
            "racer_character_index_ram": 0,
            "engine_setting_ram": 0.5,
            "engine_setting_percent_ram": 50.0,
            "character_engine_setting_ram": 0.5,
            "racer_engine_curve_ram": 0.371747,
        }


def test_backend_step_info_includes_native_engine_setting() -> None:
    backend = _Backend()

    info = backend_step_info(backend)

    assert info["engine_setting_ram"] == 0.5
    assert info["engine_setting_percent_ram"] == 50.0
    assert info["character_engine_setting_ram"] == 0.5
    assert info["racer_engine_curve_ram"] == pytest.approx(0.371747)


def test_telemetry_info_projects_race_and_player_fields() -> None:
    telemetry = make_telemetry(
        total_lap_count=3,
        game_mode_raw=2,
        game_mode_name="time_attack",
        total_racers=6,
        course_index=4,
        course_length=1_000.0,
        difficulty_raw=3,
        difficulty_name="master",
        camera_setting_raw=3,
        camera_setting_name="wide",
        race_intro_timer=12,
        menu_selected_mode_raw=7,
        menu_difficulty_state_raw=8,
        menu_difficulty_cursor_raw=9,
        menu_transition_state_raw=10,
        menu_current_ghost_type_raw=11,
        queued_game_mode_raw=12,
        race_distance=1_500.0,
        race_time_ms=45_678,
        speed_kph=1_234.5,
        energy=99.0,
        ko_star_count=2,
        lap=2,
        laps_completed=7,
        position=5,
    )

    info = telemetry_info(telemetry)

    assert info["game_mode"] == "time_attack"
    assert info["game_mode_raw"] == 2
    assert info["difficulty"] == "master"
    assert info["difficulty_raw"] == 3
    assert info["course_index"] == 4
    assert info["camera_setting"] == "wide"
    assert info["camera_setting_raw"] == 3
    assert info["race_intro_timer"] == 12
    assert info["menu_selected_mode_raw"] == 7
    assert info["menu_difficulty_state_raw"] == 8
    assert info["menu_difficulty_cursor_raw"] == 9
    assert info["menu_transition_state_raw"] == 10
    assert info["menu_current_ghost_type_raw"] == 11
    assert info["queued_game_mode_raw"] == 12
    assert info["total_lap_count"] == 3
    assert info["race_time_ms"] == 45_678
    assert info["race_distance"] == 1_500.0
    assert info["speed_kph"] == 1_234.5
    assert info["position"] == 5
    assert info["ko_star_count"] == 2
    assert info["total_racers"] == 6
    assert info["finished"] is False
    assert info["retired"] is False
    assert info["crashed"] is False
    assert info["lap"] == 2
    assert info["laps_completed"] == 1
    assert info["race_laps_completed"] == 1
    assert info["raw_laps_completed"] == 7
    assert info["episode_completion_fraction"] == pytest.approx(0.5)
    assert info["energy"] == 99.0
