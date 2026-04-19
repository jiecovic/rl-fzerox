# src/rl_fzerox/ui/watch/view/panels/records.py
from __future__ import annotations

from rl_fzerox.ui.watch.view.panels.lines import panel_divider, panel_line
from rl_fzerox.ui.watch.view.screen.theme import PALETTE, Color
from rl_fzerox.ui.watch.view.screen.types import PanelLine, PanelSection, StatusIcon


def track_record_sections(
    *,
    current_info: dict[str, object],
    track_pool_records: tuple[dict[str, object], ...],
    best_finish_times: dict[str, int],
) -> tuple[PanelSection, ...]:
    records = track_pool_records or _current_track_record_pool(current_info)
    if not records and not best_finish_times:
        return ()
    lines: list[PanelLine] = []
    for index, record in enumerate(records):
        if index > 0:
            lines.append(panel_divider())
        lines.extend(_track_record_pool_lines(record, best_finish_times=best_finish_times))
    if not lines:
        return ()
    return (PanelSection(title="Records", lines=lines),)


def _current_track_record_pool(info: dict[str, object]) -> tuple[dict[str, object], ...]:
    if (
        _track_best_key(info) is None
        and _optional_int_info(info, "track_non_agg_best_time_ms") is None
    ):
        return ()
    return (dict(info),)


def _track_record_pool_lines(
    record: dict[str, object],
    *,
    best_finish_times: dict[str, int],
) -> tuple[PanelLine, ...]:
    watch_best = _watch_best_time_ms(record, best_finish_times)
    best_time = _optional_int_info(record, "track_non_agg_best_time_ms")
    worst_time = _optional_int_info(record, "track_non_agg_worst_time_ms")
    status_icon, status_color = _track_record_status(
        watch_best_ms=watch_best,
        best_time_ms=best_time,
        worst_time_ms=worst_time,
    )
    status_text = _track_record_gap_text(
        watch_best_ms=watch_best,
        best_time_ms=best_time,
        worst_time_ms=worst_time,
    )
    return (
        panel_line(
            _format_track_record_label(record),
            "",
            status_color,
            heading=True,
            status_icon=status_icon,
            status_text=status_text,
        ),
        panel_line(
            "PB",
            _format_optional_compact_time(watch_best),
            status_color if watch_best is not None else PALETTE.text_muted,
        ),
        panel_line(
            "WR",
            _format_record_range(best_time, worst_time),
            PALETTE.text_primary
            if best_time is not None or worst_time is not None
            else PALETTE.text_muted,
        ),
    )


def _format_track_record_label(record: dict[str, object]) -> str:
    course_name = record.get("track_course_name")
    if isinstance(course_name, str) and course_name:
        return course_name

    display_name = record.get("track_display_name")
    if isinstance(display_name, str) and display_name:
        return _short_track_name(display_name)

    course_id = record.get("track_course_id")
    if isinstance(course_id, str) and course_id:
        return _format_mode_name(course_id)

    course_index = record.get("track_course_index", record.get("course_index"))
    if isinstance(course_index, bool):
        return "Track"
    if isinstance(course_index, int):
        return f"course {course_index}"
    return "Track"


def _short_track_name(value: str) -> str:
    suffixes = (
        " Time Attack - Blue Falcon Balanced",
        " time attack blue falcon balanced",
    )
    for suffix in suffixes:
        if value.endswith(suffix):
            return value[: -len(suffix)]
    return value


def _track_record_status(
    *,
    watch_best_ms: int | None,
    best_time_ms: int | None,
    worst_time_ms: int | None,
) -> tuple[StatusIcon, Color]:
    if watch_best_ms is None:
        return "none", PALETTE.text_muted

    range_cutoff_ms = worst_time_ms if worst_time_ms is not None else best_time_ms
    if range_cutoff_ms is None:
        return "outside", PALETTE.text_warning
    if watch_best_ms <= range_cutoff_ms:
        return "in_range", PALETTE.text_accent
    return "outside", PALETTE.text_warning


def _track_record_gap_text(
    *,
    watch_best_ms: int | None,
    best_time_ms: int | None,
    worst_time_ms: int | None,
) -> str:
    if watch_best_ms is None:
        return ""
    range_cutoff_ms = worst_time_ms if worst_time_ms is not None else best_time_ms
    if range_cutoff_ms is not None and watch_best_ms > range_cutoff_ms:
        return f"+{_format_compact_duration_ms(watch_best_ms - range_cutoff_ms)}"
    if best_time_ms is None:
        return ""
    if watch_best_ms > best_time_ms:
        return f"+{_format_compact_duration_ms(watch_best_ms - best_time_ms)}"
    return f"-{_format_compact_duration_ms(best_time_ms - watch_best_ms)}"


def _format_optional_compact_time(time_ms: int | None) -> str:
    if time_ms is None:
        return "--"
    return _format_compact_race_time_ms(time_ms)


def _format_record_range(best_time_ms: int | None, worst_time_ms: int | None) -> str:
    best = _format_optional_compact_time(best_time_ms)
    worst = _format_optional_compact_time(worst_time_ms)
    return f"{best} - {worst}"


def _format_compact_race_time_ms(race_time_ms: int) -> str:
    minutes, remainder = divmod(max(0, race_time_ms), 60_000)
    seconds, milliseconds = divmod(remainder, 1_000)
    if minutes == 0:
        return f"{seconds}.{milliseconds:03d}"
    return f"{minutes}:{seconds:02d}.{milliseconds:03d}"


def _format_compact_duration_ms(duration_ms: int) -> str:
    tenths = round(max(0, duration_ms) / 100)
    if tenths < 600:
        return f"{tenths / 10:.1f}s"
    minutes, remaining_tenths = divmod(tenths, 600)
    return f"{minutes}min {remaining_tenths / 10:.1f}s"


def _watch_best_time_ms(
    info: dict[str, object],
    best_finish_times: dict[str, int],
) -> int | None:
    track_key = _track_best_key(info)
    if track_key is None:
        return None
    return best_finish_times.get(track_key)


def _track_best_key(info: dict[str, object]) -> str | None:
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


def _optional_int_info(info: dict[str, object], key: str) -> int | None:
    value = info.get(key)
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def _format_mode_name(value: object) -> str:
    if not isinstance(value, str):
        return str(value)
    return value.replace("_", " ").title()
