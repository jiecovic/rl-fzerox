# src/rl_fzerox/ui/watch/view/panels/content/records/lines.py
from __future__ import annotations

from rl_fzerox.ui.watch.view.panels.core.lines import panel_divider, panel_line
from rl_fzerox.ui.watch.view.screen.theme import PALETTE
from rl_fzerox.ui.watch.view.screen.types import PanelLine

from .formatting import (
    format_latest_compact_time,
    format_optional_compact_time,
    format_record_range,
    format_track_record_heading,
    latest_time_color,
    track_record_gap_text,
    track_record_status,
)
from .identity import (
    has_failed_attempt,
    is_current_track_record,
    optional_int_info,
    record_course_id,
    watch_track_value,
)
from .model import RecordInfo


def record_group_lines(
    records: tuple[RecordInfo, ...],
    *,
    current_info: RecordInfo,
    best_finish_times: dict[str, int],
    latest_finish_times: dict[str, int],
    latest_finish_deltas_ms: dict[str, int],
    failed_track_attempts: frozenset[str],
) -> list[PanelLine]:
    lines: list[PanelLine] = []
    for record_index, record in enumerate(records):
        if record_index > 0:
            lines.append(panel_divider())
        lines.extend(
            track_record_pool_lines(
                record,
                current_info=current_info,
                best_finish_times=best_finish_times,
                latest_finish_times=latest_finish_times,
                latest_finish_deltas_ms=latest_finish_deltas_ms,
                failed_track_attempts=failed_track_attempts,
            )
        )
    return lines


def track_record_pool_lines(
    record: RecordInfo,
    *,
    current_info: RecordInfo,
    best_finish_times: dict[str, int],
    latest_finish_times: dict[str, int],
    latest_finish_deltas_ms: dict[str, int],
    failed_track_attempts: frozenset[str],
) -> tuple[PanelLine, ...]:
    is_current_track = is_current_track_record(record, current_info)
    watch_best = watch_track_value(record, best_finish_times)
    watch_latest = watch_track_value(record, latest_finish_times)
    watch_latest_delta = watch_track_value(record, latest_finish_deltas_ms)
    failed_attempt = has_failed_attempt(record, failed_track_attempts) and watch_best is None
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
    return (
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
            "PB",
            format_optional_compact_time(watch_best),
            status_color if watch_best is not None else PALETTE.text_muted,
        ),
        panel_line(
            "Latest",
            format_latest_compact_time(
                watch_latest,
                watch_best,
                latest_delta_ms=watch_latest_delta,
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
            "WR",
            format_record_range(best_time, worst_time),
            PALETTE.text_primary
            if best_time is not None or worst_time is not None
            else PALETTE.text_muted,
        ),
    )
