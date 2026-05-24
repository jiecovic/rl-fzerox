# src/rl_fzerox/ui/watch/runtime/course_navigation.py
from __future__ import annotations

from rl_fzerox.core.runtime_spec.schema import TrackSamplingEntryConfig


def current_watch_course_id(info: dict[str, object]) -> str | None:
    course_id = info.get("track_course_id")
    return course_id if isinstance(course_id, str) and course_id else None


def watch_sequential_course_ids(
    entries: tuple[TrackSamplingEntryConfig, ...],
) -> tuple[str, ...]:
    seen_course_keys: set[str] = set()
    ordered_course_ids: list[str] = []
    for entry in entries:
        course_key = watch_entry_course_key(entry)
        if course_key in seen_course_keys:
            continue
        seen_course_keys.add(course_key)
        if entry.course_id:
            ordered_course_ids.append(entry.course_id)
    return tuple(ordered_course_ids)


def watch_entry_course_key(entry: TrackSamplingEntryConfig) -> str:
    if entry.course_id:
        return f"course_id:{entry.course_id}"
    if entry.course_ref:
        return f"course_ref:{entry.course_ref}"
    if entry.course_index is not None:
        return f"course_index:{int(entry.course_index)}"
    return f"entry:{entry.id}"


def adjacent_watch_course_id(
    *,
    current_course_id: str | None,
    ordered_course_ids: tuple[str, ...],
    offset: int,
) -> str | None:
    if current_course_id is None:
        return None
    if not ordered_course_ids:
        return current_course_id
    try:
        current_index = ordered_course_ids.index(current_course_id)
    except ValueError:
        return current_course_id
    next_index = (current_index + offset) % len(ordered_course_ids)
    return ordered_course_ids[next_index]


def sync_locked_course_info(
    *,
    info: dict[str, object],
    reset_info: dict[str, object],
    locked_reset_course_id: str | None,
) -> None:
    if locked_reset_course_id is None:
        info.pop("track_sampling_locked_course_id", None)
        reset_info.pop("track_sampling_locked_course_id", None)
        return
    info["track_sampling_locked_course_id"] = locked_reset_course_id
    reset_info["track_sampling_locked_course_id"] = locked_reset_course_id
