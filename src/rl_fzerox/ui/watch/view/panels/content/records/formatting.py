# src/rl_fzerox/ui/watch/view/panels/content/records/formatting.py
from __future__ import annotations

from rl_fzerox.ui.watch.view.screen.theme import PALETTE, Color
from rl_fzerox.ui.watch.view.screen.types import StatusIcon

from .model import RecordInfo


def format_cup_label(cup: str | None) -> str:
    if cup is None:
        return "Other"
    label = format_mode_name(cup)
    return label if label.lower().endswith("cup") else f"{label} Cup"


def format_track_record_heading(record: RecordInfo, *, is_current_track: bool) -> str:
    label = format_track_record_label(record)
    return f"> {label}" if is_current_track else label


def format_track_record_label(record: RecordInfo) -> str:
    course_name = record.get("track_course_name")
    if isinstance(course_name, str) and course_name:
        return course_name

    display_name = record.get("track_display_name")
    if isinstance(display_name, str) and display_name:
        return _short_track_name(display_name)

    course_id = record.get("track_course_id")
    if isinstance(course_id, str) and course_id:
        return format_mode_name(course_id)

    course_index = record.get("track_course_index", record.get("course_index"))
    if isinstance(course_index, bool):
        return "Track"
    if isinstance(course_index, int):
        return f"course {course_index}"
    return "Track"


def track_record_status(
    *,
    watch_best_ms: int | None,
    best_time_ms: int | None,
    worst_time_ms: int | None,
    failed_attempt: bool,
) -> tuple[StatusIcon, Color]:
    if failed_attempt:
        return "outside", PALETTE.text_warning
    if watch_best_ms is None:
        return "none", PALETTE.text_muted

    range_cutoff_ms = worst_time_ms if worst_time_ms is not None else best_time_ms
    if range_cutoff_ms is None:
        return "outside", PALETTE.text_warning
    if watch_best_ms <= range_cutoff_ms:
        return "in_range", PALETTE.text_accent
    return "outside", PALETTE.text_warning


def track_record_gap_text(
    *,
    watch_best_ms: int | None,
    best_time_ms: int | None,
    worst_time_ms: int | None,
    failed_attempt: bool,
) -> str:
    if failed_attempt:
        return "FAILED"
    if watch_best_ms is None:
        return ""
    range_cutoff_ms = worst_time_ms if worst_time_ms is not None else best_time_ms
    if range_cutoff_ms is not None and watch_best_ms > range_cutoff_ms:
        return f"+{format_compact_duration_ms(watch_best_ms - range_cutoff_ms)}"
    if best_time_ms is None:
        return ""
    if watch_best_ms > best_time_ms:
        return f"+{format_compact_duration_ms(watch_best_ms - best_time_ms)}"
    return f"-{format_compact_duration_ms(best_time_ms - watch_best_ms)}"


def format_optional_compact_time(time_ms: int | None) -> str:
    if time_ms is None:
        return "--"
    return format_compact_race_time_ms(time_ms)


def format_latest_compact_time(
    latest_time_ms: int | None,
    best_time_ms: int | None,
    *,
    latest_delta_ms: int | None,
    failed_attempt: bool,
) -> str:
    if latest_time_ms is None:
        return "failed" if failed_attempt else "--"
    latest = format_compact_race_time_ms(latest_time_ms)
    if best_time_ms is None:
        return latest

    # Delta is latest finish time minus the comparison PB. Negative is faster.
    delta_ms = latest_delta_ms if latest_delta_ms is not None else latest_time_ms - best_time_ms
    if delta_ms == 0:
        return latest
    sign = "+" if delta_ms > 0 else "-"
    return f"{latest} ({sign}{format_compact_duration_ms(abs(delta_ms))})"


def latest_time_color(
    *,
    latest_time_ms: int | None,
    best_time_ms: int | None,
    latest_delta_ms: int | None,
    failed_attempt: bool,
) -> Color:
    if failed_attempt:
        return PALETTE.text_warning
    if latest_time_ms is None:
        return PALETTE.text_muted
    if latest_delta_ms is not None:
        return PALETTE.text_warning if latest_delta_ms > 0 else PALETTE.text_accent
    if best_time_ms is not None and latest_time_ms > best_time_ms:
        return PALETTE.text_warning
    return PALETTE.text_accent


def format_record_range(best_time_ms: int | None, worst_time_ms: int | None) -> str:
    best = format_optional_compact_time(best_time_ms)
    worst = format_optional_compact_time(worst_time_ms)
    return f"{best} - {worst}"


def format_compact_duration_ms(duration_ms: int) -> str:
    tenths = round(max(0, duration_ms) / 100)
    if tenths < 600:
        return f"{tenths / 10:.1f}s"
    minutes, remaining_tenths = divmod(tenths, 600)
    return f"{minutes}min {remaining_tenths / 10:.1f}s"


def format_compact_race_time_ms(race_time_ms: int) -> str:
    minutes, remainder = divmod(max(0, race_time_ms), 60_000)
    seconds, milliseconds = divmod(remainder, 1_000)
    if minutes == 0:
        return f"{seconds}.{milliseconds:03d}"
    return f"{minutes}:{seconds:02d}.{milliseconds:03d}"


def format_mode_name(value: object) -> str:
    if not isinstance(value, str):
        return str(value)
    return value.replace("_", " ").title()


def _short_track_name(value: str) -> str:
    suffixes = (
        " Time Attack - Blue Falcon Balanced",
        " time attack blue falcon balanced",
    )
    for suffix in suffixes:
        if value.endswith(suffix):
            return value[: -len(suffix)]
    return value
