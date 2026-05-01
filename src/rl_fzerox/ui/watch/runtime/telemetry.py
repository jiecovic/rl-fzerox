# src/rl_fzerox/ui/watch/runtime/telemetry.py
from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

from fzerox_emulator import FZeroXTelemetry, PlayerTelemetry


class TelemetryReader(Protocol):
    def try_read_telemetry(self) -> FZeroXTelemetry | None: ...


def _read_live_telemetry(emulator: TelemetryReader) -> FZeroXTelemetry | None:
    return emulator.try_read_telemetry()


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
        damage_rumble_counter=_mapping_int(player_value, "damage_rumble_counter"),
        reverse_timer=_mapping_int(player_value, "reverse_timer"),
        race_distance=_mapping_float(player_value, "race_distance"),
        lap_distance=_mapping_float(player_value, "lap_distance"),
        race_time_ms=_mapping_int(player_value, "race_time_ms"),
        lap=_mapping_int(player_value, "lap"),
        laps_completed=_mapping_int(player_value, "laps_completed"),
        position=_mapping_int(player_value, "position"),
        segment_index=_mapping_optional_int(player_value, "segment_index"),
        segment_t=_mapping_float(player_value, "segment_t"),
        segment_length_proportion=_mapping_float(player_value, "segment_length_proportion"),
        local_lateral_velocity=_mapping_float(player_value, "local_lateral_velocity"),
        signed_lateral_offset=_mapping_float(player_value, "signed_lateral_offset"),
        lateral_distance=_mapping_float(player_value, "lateral_distance"),
        lateral_displacement_magnitude=_mapping_float(
            player_value,
            "lateral_displacement_magnitude",
        ),
        current_radius_left=_mapping_float(player_value, "current_radius_left"),
        current_radius_right=_mapping_float(player_value, "current_radius_right"),
        height_above_ground=_mapping_float(player_value, "height_above_ground"),
        velocity_magnitude=_mapping_float(player_value, "velocity_magnitude"),
        acceleration_magnitude=_mapping_float(player_value, "acceleration_magnitude"),
        acceleration_force=_mapping_float(player_value, "acceleration_force"),
        drift_attack_force=_mapping_float(player_value, "drift_attack_force"),
        collision_mass=_mapping_float(player_value, "collision_mass"),
        machine_body_stat=_mapping_int(player_value, "machine_body_stat"),
        machine_boost_stat=_mapping_int(player_value, "machine_boost_stat"),
        machine_grip_stat=_mapping_int(player_value, "machine_grip_stat"),
        machine_weight=_mapping_int(player_value, "machine_weight"),
        engine_setting=_mapping_float(player_value, "engine_setting"),
    )
    return FZeroXTelemetry(
        total_lap_count=_mapping_int(data, "total_lap_count"),
        game_mode_raw=_mapping_int(data, "game_mode_raw"),
        game_mode_name=_mapping_str(data, "game_mode_name"),
        in_race_mode=_mapping_bool(data, "in_race_mode"),
        total_racers=_mapping_int(data, "total_racers"),
        course_index=_mapping_int(data, "course_index"),
        course_segment_count=_mapping_int(data, "course_segment_count"),
        course_length=_mapping_float(data, "course_length"),
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


def _mapping_optional_int(data: Mapping[str, object], key: str) -> int | None:
    value = data.get(key)
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int | float):
        return int(value)
    return None


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
