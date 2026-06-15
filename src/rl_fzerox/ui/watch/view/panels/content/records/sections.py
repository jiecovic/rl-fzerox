# src/rl_fzerox/ui/watch/view/panels/content/records/sections.py
from __future__ import annotations

from rl_fzerox.ui.watch.records import TrackRecordBook, record_difficulty
from rl_fzerox.ui.watch.view.screen.types import PanelSection

from .formatting import format_cup_label
from .grouping import record_groups, should_split_cup_sections
from .identity import current_track_record_pool, track_best_key
from .lines import record_group_lines
from .model import RecordInfo


def track_record_sections(
    *,
    current_info: RecordInfo,
    track_pool_records: tuple[RecordInfo, ...],
    track_record_book: TrackRecordBook,
    allow_course_jumps: bool = True,
) -> tuple[PanelSection, ...]:
    records = _unique_course_records(
        _records_for_selected_difficulty(
            track_pool_records or current_track_record_pool(current_info),
            current_info=current_info,
        )
    )
    if not records and track_record_book.is_empty:
        return ()
    groups = record_groups(records)
    if should_split_cup_sections(groups):
        return tuple(
            PanelSection(
                title=format_cup_label(group.cup),
                lines=record_group_lines(
                    group.records,
                    current_info=current_info,
                    track_record_book=track_record_book,
                    allow_course_jumps=allow_course_jumps,
                ),
            )
            for group in groups
        )

    lines = record_group_lines(
        records,
        current_info=current_info,
        track_record_book=track_record_book,
        allow_course_jumps=allow_course_jumps,
    )
    if not lines:
        return ()
    return (PanelSection(title="Records", lines=lines),)


def _records_for_selected_difficulty(
    records: tuple[RecordInfo, ...],
    *,
    current_info: RecordInfo,
) -> tuple[RecordInfo, ...]:
    difficulty = _selected_difficulty(current_info)
    if difficulty is None:
        return records
    exact_records = tuple(record for record in records if record_difficulty(record) == difficulty)
    if exact_records:
        return exact_records
    return tuple(record for record in records if record_difficulty(record) is None)


def _selected_difficulty(current_info: RecordInfo) -> str | None:
    selected = current_info.get("watch_selected_gp_difficulty")
    if isinstance(selected, str) and selected:
        return selected
    return record_difficulty(current_info)


def _unique_course_records(records: tuple[RecordInfo, ...]) -> tuple[RecordInfo, ...]:
    unique: list[RecordInfo] = []
    seen: set[str] = set()
    for record in records:
        record_key = track_best_key(record)
        if record_key is None:
            unique.append(record)
            continue
        if record_key in seen:
            continue
        seen.add(record_key)
        unique.append(record)
    return tuple(unique)
