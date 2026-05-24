# src/rl_fzerox/core/envs/engine/reset/gp_race.py
from __future__ import annotations

from fzerox_emulator import EmulatorBackend, FZeroXTelemetry
from rl_fzerox.core.envs.engine.info import read_live_telemetry
from rl_fzerox.core.envs.engine.reset.tracks import SelectedTrack


def retarget_gp_race_baseline(
    *,
    backend: EmulatorBackend,
    selected_track: SelectedTrack,
    telemetry: FZeroXTelemetry | None,
    info: dict[str, object],
) -> FZeroXTelemetry | None:
    """Patch engine only on one exact GP course+vehicle baseline."""

    if not _needs_gp_race_retarget(selected_track):
        return telemetry
    if selected_track.course_index is None:
        raise ValueError("gp_race retarget requires selected_track.course_index")
    if selected_track.vehicle is None:
        raise ValueError("gp_race retarget requires selected_track.vehicle")
    if selected_track.engine_setting_raw_value is None:
        raise ValueError("gp_race retarget requires selected_track.engine_setting_raw_value")
    if selected_track.course_index != selected_track.source_course_index:
        raise RuntimeError("gp baseline course mismatch; stale exact baseline")
    if selected_track.vehicle != selected_track.source_vehicle:
        raise RuntimeError("gp baseline vehicle mismatch; stale exact baseline")
    if (
        selected_track.gp_difficulty is not None
        and selected_track.source_gp_difficulty is not None
        and selected_track.gp_difficulty != selected_track.source_gp_difficulty
    ):
        raise RuntimeError("gp baseline difficulty mismatch; stale exact baseline")

    backend.patch_engine_settings(
        mode="gp_race",
        engine_setting_raw_value=int(selected_track.engine_setting_raw_value),
    )
    info["track_gp_race_retargeted"] = True
    info["track_source_vehicle"] = selected_track.source_vehicle
    info["track_source_engine_setting"] = selected_track.source_engine_setting
    info["track_source_engine_setting_raw_value"] = selected_track.source_engine_setting_raw_value
    return read_live_telemetry(backend) or telemetry


def _needs_gp_race_retarget(selected_track: SelectedTrack) -> bool:
    if selected_track.mode != "gp_race":
        return False
    return bool(
        selected_track.engine_setting_raw_value != selected_track.source_engine_setting_raw_value
    )
