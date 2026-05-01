# src/rl_fzerox/ui/watch/view/panels/content/records/identity.py
from __future__ import annotations

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
    track_key = track_best_key(record)
    return track_key is not None and track_key in failed_track_attempts


def record_course_id(record: RecordInfo) -> str | None:
    course_id = record.get("track_course_id")
    if isinstance(course_id, str) and course_id:
        return course_id
    return None


def is_current_track_record(
    record: RecordInfo,
    current_info: RecordInfo,
) -> bool:
    current_key = track_best_key(current_info)
    record_key = track_best_key(record)
    if current_key is not None and record_key is not None:
        return current_key == record_key

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
    track_key = track_best_key(info)
    if track_key is None:
        return None
    return values.get(track_key)


def track_best_key(info: RecordInfo) -> str | None:
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


def optional_int_info(info: RecordInfo, key: str) -> int | None:
    value = info.get(key)
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None
