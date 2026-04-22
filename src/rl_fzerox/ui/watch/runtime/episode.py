# src/rl_fzerox/ui/watch/runtime/episode.py
from __future__ import annotations

from fzerox_emulator import FZeroXTelemetry

TrackFinishTimes = dict[str, int]
TrackBestFinishTimes = TrackFinishTimes
TrackLatestFinishDeltas = dict[str, int]


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
    for key in ("track_id", "track_display_name"):
        value = info.get(key)
        if isinstance(value, str) and value:
            return value
    value = info.get("track_course_index", info.get("course_index"))
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return f"course:{value}"
    return None
