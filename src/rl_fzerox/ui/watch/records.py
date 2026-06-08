# src/rl_fzerox/ui/watch/records.py
from __future__ import annotations

from collections.abc import Mapping

from rl_fzerox.core.domain.race_difficulty import is_race_difficulty_name
from rl_fzerox.core.domain.x_cup import X_CUP_COURSE

TrackRecordKey = str


def track_record_key(info: Mapping[str, object]) -> TrackRecordKey | None:
    """Return the watch-local record key for one course/difficulty attempt."""

    base_key = base_track_record_key(info)
    if base_key is None:
        return None
    difficulty = record_difficulty(info)
    if difficulty is None:
        return base_key
    return f"{base_key}#difficulty={difficulty}"


def track_record_lookup_keys(info: Mapping[str, object]) -> tuple[TrackRecordKey, ...]:
    """Return candidate keys from most specific to legacy/plain lookup forms."""

    base_key = base_track_record_key(info)
    if base_key is None:
        return ()
    difficulty = record_difficulty(info)
    keys = [base_key if difficulty is None else f"{base_key}#difficulty={difficulty}"]
    keys.extend(_legacy_track_keys(info))
    keys.append(base_key)
    return tuple(dict.fromkeys(keys))


def base_track_record_key(info: Mapping[str, object]) -> TrackRecordKey | None:
    if _is_generated_x_cup_record(info):
        generated_hash = info.get("track_generated_course_hash")
        if isinstance(generated_hash, str) and generated_hash:
            return f"x_cup:{generated_hash}"

    course_id = info.get("track_course_id")
    if isinstance(course_id, str) and course_id:
        return course_id

    value = info.get("track_course_index", info.get("course_index"))
    if isinstance(value, int) and not isinstance(value, bool):
        return f"course:{value}"

    for key in ("track_id", "track_display_name"):
        value = info.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def record_difficulty(info: Mapping[str, object]) -> str | None:
    for key in (
        "track_gp_difficulty",
        "track_source_gp_difficulty",
        "gp_difficulty",
        "source_gp_difficulty",
    ):
        difficulty = _valid_difficulty(info.get(key))
        if difficulty is not None:
            return difficulty

    if info.get("track_mode") == X_CUP_COURSE.race_mode:
        for key in ("difficulty_name", "difficulty"):
            difficulty = _valid_difficulty(info.get(key))
            if difficulty is not None:
                return difficulty
    return None


def _valid_difficulty(value: object) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    return value if is_race_difficulty_name(value) else None


def _legacy_track_keys(info: Mapping[str, object]) -> tuple[str, ...]:
    keys: list[str] = []

    course_id = info.get("track_course_id")
    if isinstance(course_id, str) and course_id:
        keys.append(course_id)

    course_index = info.get("track_course_index", info.get("course_index"))
    if isinstance(course_index, int) and not isinstance(course_index, bool):
        keys.append(f"course:{course_index}")

    for field_name in ("track_id", "track_display_name"):
        value = info.get(field_name)
        if isinstance(value, str) and value:
            keys.append(value)

    return tuple(keys)


def _is_generated_x_cup_record(info: Mapping[str, object]) -> bool:
    if info.get("track_generated_course_kind") == X_CUP_COURSE.generated_kind:
        return True

    for key in ("track_runtime_course_key", "track_reset_course_key", "track_course_id"):
        value = info.get(key)
        if isinstance(value, str) and value.startswith(X_CUP_COURSE.id_prefix):
            return True

    course_index = info.get("track_course_index", info.get("course_index"))
    return (
        isinstance(course_index, int)
        and not isinstance(course_index, bool)
        and (course_index == X_CUP_COURSE.course_index)
    )
