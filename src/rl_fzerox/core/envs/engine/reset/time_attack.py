# src/rl_fzerox/core/envs/engine/reset/time_attack.py
from __future__ import annotations

from fzerox_emulator import EmulatorBackend, FZeroXTelemetry
from rl_fzerox.core.config.vehicle_catalog import vehicle_by_id
from rl_fzerox.core.training.runs.baseline_race_start import RACE_DEFAULTS

from ..info import read_live_telemetry
from .tracks import SelectedTrack


def retarget_time_attack_baseline(
    *,
    backend: EmulatorBackend,
    selected_track: SelectedTrack,
    telemetry: FZeroXTelemetry | None,
    info: dict[str, object],
) -> FZeroXTelemetry | None:
    """Patch one loaded time-attack baseline to the selected track setup."""

    if not _needs_time_attack_retarget(selected_track):
        return telemetry
    if selected_track.course_index is None:
        raise ValueError("time-attack retarget requires selected_track.course_index")
    if selected_track.vehicle is None:
        raise ValueError("time-attack retarget requires selected_track.vehicle")
    if selected_track.engine_setting_raw_value is None:
        raise ValueError("time-attack retarget requires selected_track.engine_setting_raw_value")

    vehicle = vehicle_by_id(selected_track.vehicle)
    total_lap_count = (
        int(telemetry.total_lap_count) if telemetry is not None else RACE_DEFAULTS.lap_count
    )
    patch_kwargs = {
        "mode": "time_attack",
        "course_index": int(selected_track.course_index),
        "character_index": int(vehicle.character_index),
        "machine_skin_index": int(RACE_DEFAULTS.machine_skin_index),
        "engine_setting_raw_value": int(selected_track.engine_setting_raw_value),
        "total_lap_count": total_lap_count,
    }
    backend.patch_machine_settings(**patch_kwargs)
    backend.patch_race_start_setup(**patch_kwargs)
    backend.force_race_reinit(mode="time_attack")
    info["track_time_attack_retargeted"] = True
    info["track_source_vehicle"] = selected_track.source_vehicle
    info["track_source_engine_setting"] = selected_track.source_engine_setting
    info["track_source_engine_setting_raw_value"] = selected_track.source_engine_setting_raw_value
    return read_live_telemetry(backend) or telemetry


def _needs_time_attack_retarget(selected_track: SelectedTrack) -> bool:
    if selected_track.mode != "time_attack":
        return False
    return bool(
        selected_track.course_index != selected_track.source_course_index
        or selected_track.vehicle != selected_track.source_vehicle
        or selected_track.engine_setting_raw_value != selected_track.source_engine_setting_raw_value
    )
