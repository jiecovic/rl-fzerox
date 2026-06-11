# src/rl_fzerox/ui/watch/view/panels/content/records/lines.py
from __future__ import annotations

from rl_fzerox.ui.watch.records import TrackRecordBook
from rl_fzerox.ui.watch.view.panels.core.lines import panel_divider, panel_line
from rl_fzerox.ui.watch.view.screen.theme import PALETTE
from rl_fzerox.ui.watch.view.screen.types import PanelLine

from .formatting import (
    format_attempt_stats,
    format_best_position,
    format_best_time,
    format_latest_compact_time,
    format_record_range,
    format_track_record_heading,
    latest_time_color,
    track_record_gap_text,
    track_record_status,
)
from .identity import (
    is_current_track_record,
    optional_int_info,
    record_course_id,
)
from .model import RecordInfo


def record_group_lines(
    records: tuple[RecordInfo, ...],
    *,
    current_info: RecordInfo,
    track_record_book: TrackRecordBook,
) -> list[PanelLine]:
    lines: list[PanelLine] = []
    for record_index, record in enumerate(records):
        if record_index > 0:
            lines.append(panel_divider())
        lines.extend(
            track_record_pool_lines(
                record,
                current_info=current_info,
                track_record_book=track_record_book,
            )
        )
    return lines


def track_record_pool_lines(
    record: RecordInfo,
    *,
    current_info: RecordInfo,
    track_record_book: TrackRecordBook,
) -> tuple[PanelLine, ...]:
    is_current_track = is_current_track_record(record, current_info)
    entry = track_record_book.entry_for(record)
    watch_best_rank = None if entry is None else entry.best_finish_rank
    watch_best_rank_time = None if entry is None else entry.best_finish_rank_time_ms
    watch_best_rank_setup = None if entry is None else entry.best_finish_rank_setup
    watch_best = None if entry is None else entry.best_finish_time_ms
    watch_best_time_rank = None if entry is None else entry.best_finish_time_rank
    watch_best_time_setup = None if entry is None else entry.best_finish_time_setup
    watch_latest_rank = None if entry is None else entry.latest_finish_rank
    watch_latest = None if entry is None else entry.latest_finish_time_ms
    watch_latest_delta = None if entry is None else entry.latest_finish_delta_ms
    watch_latest_setup = None if entry is None else entry.latest_finish_setup
    watch_attempt_stats = None if entry is None else entry.attempt_stats.as_mapping()
    failed_attempt = bool(entry is not None and entry.failed_attempt and watch_best is None)
    best_time = optional_int_info(record, "track_non_agg_best_time_ms")
    worst_time = optional_int_info(record, "track_non_agg_worst_time_ms")
    status_icon, status_color = track_record_status(
        watch_best_ms=watch_best,
        best_time_ms=best_time,
        worst_time_ms=worst_time,
        failed_attempt=failed_attempt,
    )
    status_text = track_record_gap_text(
        watch_best_ms=watch_best,
        best_time_ms=best_time,
        worst_time_ms=worst_time,
        failed_attempt=failed_attempt,
    )
    lines = [
        panel_line(
            format_track_record_heading(record, is_current_track=is_current_track),
            "",
            status_color,
            heading=True,
            status_icon=status_icon,
            status_text="LIVE" if is_current_track and not status_text else status_text,
            label_color=PALETTE.text_accent if is_current_track else None,
            click_course_id=record_course_id(record),
        ),
        panel_line(
            "Best time",
            format_best_time(
                watch_best,
                rank=watch_best_time_rank if _shows_finish_rank(record, current_info) else None,
                setup=watch_best_time_setup,
            ),
            status_color if watch_best is not None else PALETTE.text_muted,
        ),
    ]
    if _shows_finish_rank(record, current_info):
        lines.append(
            panel_line(
                "Best pos",
                format_best_position(
                    watch_best_rank,
                    time_ms=watch_best_rank_time,
                    setup=watch_best_rank_setup,
                ),
                PALETTE.text_accent if watch_best_rank is not None else PALETTE.text_muted,
            )
        )
    lines.extend(
        (
            panel_line(
                "Latest",
                format_latest_compact_time(
                    watch_latest,
                    watch_best,
                    latest_delta_ms=watch_latest_delta,
                    rank=watch_latest_rank if _shows_finish_rank(record, current_info) else None,
                    setup=watch_latest_setup,
                    failed_attempt=failed_attempt,
                ),
                latest_time_color(
                    latest_time_ms=watch_latest,
                    best_time_ms=watch_best,
                    latest_delta_ms=watch_latest_delta,
                    failed_attempt=failed_attempt,
                ),
            ),
            panel_line(
                "Attempts",
                format_attempt_stats(watch_attempt_stats),
                PALETTE.text_primary if watch_attempt_stats is not None else PALETTE.text_muted,
            ),
            panel_line(
                "WR",
                format_record_range(best_time, worst_time),
                PALETTE.text_primary
                if best_time is not None or worst_time is not None
                else PALETTE.text_muted,
            ),
        )
    )
    return tuple(lines)


def _shows_finish_rank(record: RecordInfo, current_info: RecordInfo) -> bool:
    return record.get("track_mode") == "gp_race" or current_info.get("track_mode") == "gp_race"
