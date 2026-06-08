# src/rl_fzerox/ui/watch/view/panels/content/records/identity.py
from __future__ import annotations

from rl_fzerox.ui.watch.records import (
    record_difficulty,
    track_record_key,
    track_record_lookup_keys,
)

from .model import RecordInfo


def current_track_record_pool(info: RecordInfo) -> tuple[RecordInfo, ...]:
    has_best_time = (
        track_best_key(info) is not None
        or optional_int_info(info, "track_non_agg_best_time_ms") is not None
    )
    if not has_best_time:
        return ()
    return (dict(info),)


def has_failed_attempt(
    record: RecordInfo,
    failed_track_attempts: frozenset[str],
) -> bool:
    return any(track_key in failed_track_attempts for track_key in track_record_lookup_keys(record))


def record_course_id(record: RecordInfo) -> str | None:
    reset_target_key = record.get("track_reset_target_key")
    if isinstance(reset_target_key, str) and reset_target_key:
        return reset_target_key
    reset_course_key = record.get("track_reset_course_key")
    if isinstance(reset_course_key, str) and reset_course_key:
        return reset_course_key
    runtime_course_key = record.get("track_runtime_course_key")
    if isinstance(runtime_course_key, str) and runtime_course_key:
        return runtime_course_key
    course_id = record.get("track_course_id")
    if isinstance(course_id, str) and course_id:
        return course_id
    return None


def is_current_track_record(
    record: RecordInfo,
    current_info: RecordInfo,
) -> bool:
    current_key = track_record_key(current_info)
    record_key = track_record_key(record)
    if current_key is not None and record_key is not None:
        return current_key == record_key

    current_difficulty = record_difficulty(current_info)
    record_difficulty_value = record_difficulty(record)
    if (
        current_difficulty is not None
        and record_difficulty_value is not None
        and current_difficulty != record_difficulty_value
    ):
        return False

    current_course_id = current_info.get("track_course_id")
    record_course_id = record.get("track_course_id")
    if (
        isinstance(current_course_id, str)
        and current_course_id
        and current_course_id == record_course_id
    ):
        return True

    current_index = current_info.get("track_course_index", current_info.get("course_index"))
    record_index = record.get("track_course_index", record.get("course_index"))
    if isinstance(current_index, bool) or isinstance(record_index, bool):
        return False
    return isinstance(current_index, int) and current_index == record_index


def watch_track_value(
    info: RecordInfo,
    values: dict[str, int],
) -> int | None:
    for track_key in track_record_lookup_keys(info):
        value = values.get(track_key)
        if value is not None:
            return value
    return None


def track_best_key(info: RecordInfo) -> str | None:
    return track_record_key(info)


def optional_int_info(info: RecordInfo, key: str) -> int | None:
    value = info.get(key)
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None
