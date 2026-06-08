# src/rl_fzerox/ui/watch/runtime/episode.py
from __future__ import annotations

import math

from fzerox_emulator import FZeroXTelemetry
from rl_fzerox.core.runtime_spec.vehicle_catalog import vehicle_by_character_index
from rl_fzerox.ui.watch.records import track_record_key

TrackFinishTimes = dict[str, int]
TrackBestFinishTimes = TrackFinishTimes
TrackFinishRanks = dict[str, int]
TrackBestFinishRanks = TrackFinishRanks
TrackFinishSetup = dict[str, str | int]
TrackBestFinishSetups = dict[str, TrackFinishSetup]
TrackLatestFinishDeltas = dict[str, int]
FailedTrackAttempts = frozenset[str]


def _update_best_finish_position(
    best_finish_position: int | None,
    info: dict[str, object],
    telemetry: FZeroXTelemetry | None,
) -> int | None:
    finish_position = _successful_finish_position(info, telemetry)
    if finish_position is None:
        return best_finish_position
    if best_finish_position is None:
        return finish_position
    return min(best_finish_position, finish_position)


def _update_best_finish_times(
    best_finish_times: TrackBestFinishTimes,
    info: dict[str, object],
    telemetry: FZeroXTelemetry | None,
) -> TrackBestFinishTimes:
    """Return updated per-track best finish times for successful episodes."""

    finish_time_ms = _successful_finish_time_ms(info, telemetry)
    track_key = _track_key(info)
    if finish_time_ms is None or track_key is None:
        return best_finish_times
    current_best = best_finish_times.get(track_key)
    if current_best is not None and current_best <= finish_time_ms:
        return best_finish_times
    updated = dict(best_finish_times)
    updated[track_key] = finish_time_ms
    return updated


def _update_best_finish_time_ranks(
    best_finish_time_ranks: TrackBestFinishRanks,
    best_finish_times: TrackBestFinishTimes,
    info: dict[str, object],
    telemetry: FZeroXTelemetry | None,
) -> TrackBestFinishRanks:
    """Return the finish position attached to each per-track best-time finish."""

    finish_time_ms = _successful_finish_time_ms(info, telemetry)
    finish_position = _successful_finish_position(info, telemetry)
    track_key = _track_key(info)
    if (
        finish_time_ms is None
        or finish_position is None
        or track_key is None
        or not _is_new_best_time(best_finish_times, track_key, finish_time_ms)
    ):
        return best_finish_time_ranks
    updated = dict(best_finish_time_ranks)
    updated[track_key] = finish_position
    return updated


def _update_best_finish_time_setups(
    best_finish_time_setups: TrackBestFinishSetups,
    best_finish_times: TrackBestFinishTimes,
    info: dict[str, object],
    telemetry: FZeroXTelemetry | None,
) -> TrackBestFinishSetups:
    """Return vehicle/engine setup metadata for each per-track best-time finish."""

    finish_time_ms = _successful_finish_time_ms(info, telemetry)
    track_key = _track_key(info)
    if (
        finish_time_ms is None
        or track_key is None
        or not _is_new_best_time(best_finish_times, track_key, finish_time_ms)
    ):
        return best_finish_time_setups
    setup = _finish_setup(info, telemetry)
    if not setup:
        return best_finish_time_setups
    updated = dict(best_finish_time_setups)
    updated[track_key] = setup
    return updated


def _update_best_finish_ranks(
    best_finish_ranks: TrackBestFinishRanks,
    info: dict[str, object],
    telemetry: FZeroXTelemetry | None,
) -> TrackBestFinishRanks:
    """Return updated per-track best finish positions for successful episodes."""

    finish_position = _successful_finish_position(info, telemetry)
    track_key = _track_key(info)
    if finish_position is None or track_key is None:
        return best_finish_ranks
    current_best = best_finish_ranks.get(track_key)
    if current_best is not None and current_best <= finish_position:
        return best_finish_ranks
    updated = dict(best_finish_ranks)
    updated[track_key] = finish_position
    return updated


def _update_best_finish_rank_times(
    best_finish_rank_times: TrackBestFinishTimes,
    best_finish_ranks: TrackBestFinishRanks,
    info: dict[str, object],
    telemetry: FZeroXTelemetry | None,
) -> TrackBestFinishTimes:
    """Return the finish time attached to each per-track best-position finish."""

    finish_time_ms = _successful_finish_time_ms(info, telemetry)
    finish_position = _successful_finish_position(info, telemetry)
    track_key = _track_key(info)
    if (
        finish_time_ms is None
        or finish_position is None
        or track_key is None
        or not _is_new_best_rank(
            best_finish_ranks,
            best_finish_rank_times,
            track_key,
            finish_position,
            finish_time_ms,
        )
    ):
        return best_finish_rank_times
    updated = dict(best_finish_rank_times)
    updated[track_key] = finish_time_ms
    return updated


def _update_best_finish_rank_setups(
    best_finish_rank_setups: TrackBestFinishSetups,
    best_finish_rank_times: TrackBestFinishTimes,
    best_finish_ranks: TrackBestFinishRanks,
    info: dict[str, object],
    telemetry: FZeroXTelemetry | None,
) -> TrackBestFinishSetups:
    """Return vehicle/engine setup metadata for each per-track best-position finish."""

    finish_time_ms = _successful_finish_time_ms(info, telemetry)
    finish_position = _successful_finish_position(info, telemetry)
    track_key = _track_key(info)
    if (
        finish_time_ms is None
        or finish_position is None
        or track_key is None
        or not _is_new_best_rank(
            best_finish_ranks,
            best_finish_rank_times,
            track_key,
            finish_position,
            finish_time_ms,
        )
    ):
        return best_finish_rank_setups
    setup = _finish_setup(info, telemetry)
    if not setup:
        return best_finish_rank_setups
    updated = dict(best_finish_rank_setups)
    updated[track_key] = setup
    return updated


def _update_latest_finish_times(
    latest_finish_times: TrackFinishTimes,
    info: dict[str, object],
    telemetry: FZeroXTelemetry | None,
) -> TrackFinishTimes:
    """Return updated per-track latest finish times for successful episodes."""

    finish_time_ms = _successful_finish_time_ms(info, telemetry)
    track_key = _track_key(info)
    if finish_time_ms is None or track_key is None:
        return latest_finish_times
    updated = dict(latest_finish_times)
    updated[track_key] = finish_time_ms
    return updated


def _update_latest_finish_deltas_ms(
    latest_finish_deltas_ms: TrackLatestFinishDeltas,
    best_finish_times: TrackBestFinishTimes,
    info: dict[str, object],
    telemetry: FZeroXTelemetry | None,
) -> TrackLatestFinishDeltas:
    """Return latest finish gaps against the PB that existed before this finish."""

    finish_time_ms = _successful_finish_time_ms(info, telemetry)
    track_key = _track_key(info)
    if finish_time_ms is None or track_key is None:
        return latest_finish_deltas_ms

    updated = dict(latest_finish_deltas_ms)
    previous_best = best_finish_times.get(track_key)
    if previous_best is None:
        updated.pop(track_key, None)
    else:
        updated[track_key] = finish_time_ms - previous_best
    return updated


def _update_failed_track_attempts(
    failed_track_attempts: FailedTrackAttempts,
    info: dict[str, object],
    *,
    episode_done: bool,
) -> FailedTrackAttempts:
    """Track courses that ended in watch without a successful finish."""

    if not episode_done:
        return failed_track_attempts
    track_key = _track_key(info)
    if track_key is None:
        return failed_track_attempts

    updated = set(failed_track_attempts)
    if info.get("termination_reason") == "finished":
        updated.discard(track_key)
    else:
        updated.add(track_key)
    return frozenset(updated)


def _successful_finish_position(
    info: dict[str, object],
    telemetry: FZeroXTelemetry | None,
) -> int | None:
    if info.get("termination_reason") != "finished":
        return None

    raw_position: object
    if telemetry is not None:
        raw_position = telemetry.player.position
    else:
        raw_position = info.get("position")
    if isinstance(raw_position, bool) or not isinstance(raw_position, int):
        return None
    if raw_position <= 0:
        return None
    return raw_position


def _successful_finish_time_ms(
    info: dict[str, object],
    telemetry: FZeroXTelemetry | None,
) -> int | None:
    if info.get("termination_reason") != "finished":
        return None

    raw_time: object
    if telemetry is not None:
        raw_time = telemetry.player.race_time_ms
    else:
        raw_time = info.get("race_time_ms")
    if isinstance(raw_time, bool) or not isinstance(raw_time, int):
        return None
    if raw_time <= 0:
        return None
    return raw_time


def _track_key(info: dict[str, object]) -> str | None:
    return track_record_key(info)


def _is_new_best_time(
    best_finish_times: TrackBestFinishTimes,
    track_key: str,
    finish_time_ms: int,
) -> bool:
    current_best = best_finish_times.get(track_key)
    return current_best is None or finish_time_ms < current_best


def _is_new_best_rank(
    best_finish_ranks: TrackBestFinishRanks,
    best_finish_rank_times: TrackBestFinishTimes,
    track_key: str,
    finish_position: int,
    finish_time_ms: int,
) -> bool:
    current_rank = best_finish_ranks.get(track_key)
    if current_rank is None or finish_position < current_rank:
        return True
    if finish_position != current_rank:
        return False
    current_time = best_finish_rank_times.get(track_key)
    return current_time is None or finish_time_ms < current_time


def _finish_setup(
    info: dict[str, object],
    telemetry: FZeroXTelemetry | None,
) -> TrackFinishSetup:
    setup: TrackFinishSetup = {}
    vehicle_name, vehicle_id = _finish_vehicle(info, telemetry)
    engine_raw = _finish_engine_setting_raw(info, telemetry)
    if vehicle_name is not None:
        setup["vehicle_name"] = vehicle_name
    if vehicle_id is not None:
        setup["vehicle"] = vehicle_id
    if engine_raw is not None:
        setup["engine_setting_raw_value"] = engine_raw
    return setup


def _finish_vehicle(
    info: dict[str, object],
    telemetry: FZeroXTelemetry | None,
) -> tuple[str | None, str | None]:
    if telemetry is not None:
        character_index = _int_setup_value(
            getattr(telemetry.player, "machine_character_index", None)
        )
        if character_index is not None:
            vehicle = vehicle_by_character_index(character_index)
            if vehicle is not None:
                return vehicle.display_name, vehicle.id

    for key in ("track_vehicle_name", "vehicle_name"):
        value = info.get(key)
        if isinstance(value, str) and value:
            vehicle_id = _string_info(info, "track_vehicle") or _string_info(info, "vehicle")
            return value, vehicle_id
    vehicle_id = _string_info(info, "track_vehicle") or _string_info(info, "vehicle")
    return vehicle_id, vehicle_id


def _finish_engine_setting_raw(
    info: dict[str, object],
    telemetry: FZeroXTelemetry | None,
) -> int | None:
    if telemetry is not None:
        engine_setting = telemetry.player.engine_setting
        if math.isfinite(float(engine_setting)):
            raw_value = round(float(engine_setting) * 100.0)
            if 0 <= raw_value <= 100:
                return raw_value

    for key in (
        "engine_setting_raw_value",
        "track_engine_setting_raw_value",
        "engine_setting_percent_ram",
    ):
        raw_value = _int_setup_value(info.get(key))
        if raw_value is not None and 0 <= raw_value <= 100:
            return raw_value
    return None


def _string_info(info: dict[str, object], key: str) -> str | None:
    value = info.get(key)
    return value if isinstance(value, str) and value else None


def _int_setup_value(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        if not math.isfinite(float(value)):
            return None
        return int(round(value))
    return None
