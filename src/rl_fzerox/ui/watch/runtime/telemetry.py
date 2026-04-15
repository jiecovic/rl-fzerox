# src/rl_fzerox/ui/watch/runtime/telemetry.py
from __future__ import annotations

from collections.abc import Mapping

from fzerox_emulator import FZeroXTelemetry, PlayerTelemetry


def _telemetry_to_data(telemetry: FZeroXTelemetry | None) -> dict[str, object] | None:
    if telemetry is None:
        return None
    return dict(telemetry.to_dict())


def _telemetry_from_data(data: dict[str, object] | None) -> FZeroXTelemetry | None:
    if data is None:
        return None
    player_value = data.get("player")
    if not isinstance(player_value, Mapping):
        return None
    player = PlayerTelemetry(
        state_flags=_mapping_int(player_value, "state_flags"),
        speed_kph=_mapping_float(player_value, "speed_kph"),
        energy=_mapping_float(player_value, "energy"),
        max_energy=_mapping_float(player_value, "max_energy"),
        boost_timer=_mapping_int(player_value, "boost_timer"),
        recoil_tilt_magnitude=_mapping_float(player_value, "recoil_tilt_magnitude"),
        reverse_timer=_mapping_int(player_value, "reverse_timer"),
        race_distance=_mapping_float(player_value, "race_distance"),
        lap_distance=_mapping_float(player_value, "lap_distance"),
        race_time_ms=_mapping_int(player_value, "race_time_ms"),
        lap=_mapping_int(player_value, "lap"),
        laps_completed=_mapping_int(player_value, "laps_completed"),
        position=_mapping_int(player_value, "position"),
    )
    return FZeroXTelemetry(
        total_lap_count=_mapping_int(data, "total_lap_count"),
        game_mode_raw=_mapping_int(data, "game_mode_raw"),
        game_mode_name=_mapping_str(data, "game_mode_name"),
        in_race_mode=_mapping_bool(data, "in_race_mode"),
        total_racers=_mapping_int(data, "total_racers"),
        course_index=_mapping_int(data, "course_index"),
        player=player,
        difficulty_raw=_mapping_int(data, "difficulty_raw"),
        difficulty_name=_mapping_str(data, "difficulty_name"),
        camera_setting_raw=_mapping_int(data, "camera_setting_raw"),
        camera_setting_name=_mapping_str(data, "camera_setting_name"),
        race_intro_timer=_mapping_int(data, "race_intro_timer"),
    )


def _mapping_int(data: Mapping[str, object], key: str) -> int:
    value = data.get(key)
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int | float):
        return int(value)
    return 0


def _mapping_float(data: Mapping[str, object], key: str) -> float:
    value = data.get(key)
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, int | float):
        return float(value)
    return 0.0


def _mapping_bool(data: Mapping[str, object], key: str) -> bool:
    value = data.get(key)
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float):
        return bool(value)
    return False


def _mapping_str(data: Mapping[str, object], key: str) -> str:
    value = data.get(key)
    if isinstance(value, str):
        return value
    return ""
