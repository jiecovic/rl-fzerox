# src/rl_fzerox/ui/watch/view/panels/content/records/sections.py
from __future__ import annotations

from rl_fzerox.ui.watch.view.screen.types import PanelSection

from .formatting import format_cup_label
from .grouping import record_groups, should_split_cup_sections
from .identity import current_track_record_pool
from .lines import record_group_lines
from .model import RecordInfo


def track_record_sections(
    *,
    current_info: RecordInfo,
    track_pool_records: tuple[RecordInfo, ...],
    best_finish_times: dict[str, int],
    latest_finish_times: dict[str, int],
    latest_finish_deltas_ms: dict[str, int],
    failed_track_attempts: frozenset[str] = frozenset(),
) -> tuple[PanelSection, ...]:
    records = track_pool_records or current_track_record_pool(current_info)
    if (
        not records
        and not best_finish_times
        and not latest_finish_times
        and not failed_track_attempts
    ):
        return ()
    groups = record_groups(records)
    if should_split_cup_sections(groups):
        return tuple(
            PanelSection(
                title=format_cup_label(group.cup),
                lines=record_group_lines(
                    group.records,
                    current_info=current_info,
                    best_finish_times=best_finish_times,
                    latest_finish_times=latest_finish_times,
                    latest_finish_deltas_ms=latest_finish_deltas_ms,
                    failed_track_attempts=failed_track_attempts,
                ),
            )
            for group in groups
        )

    lines = record_group_lines(
        records,
        current_info=current_info,
        best_finish_times=best_finish_times,
        latest_finish_times=latest_finish_times,
        latest_finish_deltas_ms=latest_finish_deltas_ms,
        failed_track_attempts=failed_track_attempts,
    )
    if not lines:
        return ()
    return (PanelSection(title="Records", lines=lines),)
