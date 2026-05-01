# src/rl_fzerox/ui/watch/view/panels/content/records/grouping.py
from __future__ import annotations

from .model import BUILT_IN_COURSES_PER_CUP, BUILT_IN_CUP_ORDER, RecordGroup, RecordInfo

__all__ = (
    "record_groups",
    "should_split_cup_sections",
)


def should_split_cup_sections(record_groups: tuple[RecordGroup, ...]) -> bool:
    return len(record_groups) > 1 and any(group.cup is not None for group in record_groups)


def record_groups(records: tuple[RecordInfo, ...]) -> tuple[RecordGroup, ...]:
    records_by_cup: dict[str | None, list[RecordInfo]] = {}
    first_index_by_cup: dict[str | None, int] = {}
    for index, record in enumerate(records):
        cup = _record_cup(record)
        records_by_cup.setdefault(cup, []).append(record)
        first_index_by_cup.setdefault(cup, index)

    groups = tuple(
        RecordGroup(
            cup=cup,
            records=tuple(group_records),
            first_index=first_index_by_cup[cup],
        )
        for cup, group_records in records_by_cup.items()
    )
    return tuple(sorted(groups, key=_record_group_sort_key))


def _record_group_sort_key(group: RecordGroup) -> tuple[int, int]:
    if group.cup in BUILT_IN_CUP_ORDER:
        return (0, BUILT_IN_CUP_ORDER.index(group.cup))
    if group.cup is None:
        return (2, group.first_index)
    return (1, group.first_index)


def _record_cup(record: RecordInfo) -> str | None:
    course_ref = record.get("track_course_ref")
    if isinstance(course_ref, str) and "/" in course_ref:
        cup = course_ref.split("/", maxsplit=1)[0].strip().lower()
        if cup:
            return cup

    course_index = record.get("track_course_index", record.get("course_index"))
    if isinstance(course_index, bool) or not isinstance(course_index, int):
        return None
    cup_index = course_index // BUILT_IN_COURSES_PER_CUP
    if 0 <= cup_index < len(BUILT_IN_CUP_ORDER):
        return BUILT_IN_CUP_ORDER[cup_index]
    return None
