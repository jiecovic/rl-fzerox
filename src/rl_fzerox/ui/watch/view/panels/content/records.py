# src/rl_fzerox/ui/watch/view/panels/content/records.py
from __future__ import annotations

from dataclasses import dataclass

from rl_fzerox.ui.watch.view.panels.core.lines import panel_divider, panel_line
from rl_fzerox.ui.watch.view.screen.theme import PALETTE, Color
from rl_fzerox.ui.watch.view.screen.types import PanelLine, PanelSection, StatusIcon


@dataclass(frozen=True, slots=True)
class _RecordGroup:
    cup: str | None
    records: tuple[dict[str, object], ...]
    first_index: int


_BUILT_IN_CUP_ORDER = ("jack", "queen", "king", "joker")
_BUILT_IN_COURSES_PER_CUP = 6


def track_record_sections(
    *,
    current_info: dict[str, object],
    track_pool_records: tuple[dict[str, object], ...],
    best_finish_times: dict[str, int],
    latest_finish_times: dict[str, int],
    latest_finish_deltas_ms: dict[str, int],
    failed_track_attempts: frozenset[str] = frozenset(),
) -> tuple[PanelSection, ...]:
    records = track_pool_records or _current_track_record_pool(current_info)
    if (
        not records
        and not best_finish_times
        and not latest_finish_times
        and not failed_track_attempts
    ):
        return ()
    record_groups = _record_groups(records)
    if _should_split_cup_sections(record_groups):
        return tuple(
            PanelSection(
                title=_format_cup_label(group.cup),
                lines=_record_group_lines(
                    group.records,
                    current_info=current_info,
                    best_finish_times=best_finish_times,
                    latest_finish_times=latest_finish_times,
                    latest_finish_deltas_ms=latest_finish_deltas_ms,
                    failed_track_attempts=failed_track_attempts,
                ),
            )
            for group in record_groups
        )

    lines = _record_group_lines(
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


def _should_split_cup_sections(record_groups: tuple[_RecordGroup, ...]) -> bool:
    return len(record_groups) > 1 and any(group.cup is not None for group in record_groups)


def _record_group_lines(
    records: tuple[dict[str, object], ...],
    *,
    current_info: dict[str, object],
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
            _track_record_pool_lines(
                record,
                current_info=current_info,
                best_finish_times=best_finish_times,
                latest_finish_times=latest_finish_times,
                latest_finish_deltas_ms=latest_finish_deltas_ms,
                failed_track_attempts=failed_track_attempts,
            )
        )
    return lines


def _record_groups(records: tuple[dict[str, object], ...]) -> tuple[_RecordGroup, ...]:
    records_by_cup: dict[str | None, list[dict[str, object]]] = {}
    first_index_by_cup: dict[str | None, int] = {}
    for index, record in enumerate(records):
        cup = _record_cup(record)
        records_by_cup.setdefault(cup, []).append(record)
        first_index_by_cup.setdefault(cup, index)

    groups = tuple(
        _RecordGroup(
            cup=cup,
            records=tuple(group_records),
            first_index=first_index_by_cup[cup],
        )
        for cup, group_records in records_by_cup.items()
    )
    return tuple(sorted(groups, key=_record_group_sort_key))


def _record_group_sort_key(group: _RecordGroup) -> tuple[int, int]:
    if group.cup in _BUILT_IN_CUP_ORDER:
        return (0, _BUILT_IN_CUP_ORDER.index(group.cup))
    if group.cup is None:
        return (2, group.first_index)
    return (1, group.first_index)


def _record_cup(record: dict[str, object]) -> str | None:
    course_ref = record.get("track_course_ref")
    if isinstance(course_ref, str) and "/" in course_ref:
        cup = course_ref.split("/", maxsplit=1)[0].strip().lower()
        if cup:
            return cup

    course_index = record.get("track_course_index", record.get("course_index"))
    if isinstance(course_index, bool) or not isinstance(course_index, int):
        return None
    cup_index = course_index // _BUILT_IN_COURSES_PER_CUP
    if 0 <= cup_index < len(_BUILT_IN_CUP_ORDER):
        return _BUILT_IN_CUP_ORDER[cup_index]
    return None


def _format_cup_label(cup: str | None) -> str:
    if cup is None:
        return "Other"
    label = _format_mode_name(cup)
    return label if label.lower().endswith("cup") else f"{label} Cup"


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
    current_info: dict[str, object],
    best_finish_times: dict[str, int],
    latest_finish_times: dict[str, int],
    latest_finish_deltas_ms: dict[str, int],
    failed_track_attempts: frozenset[str],
) -> tuple[PanelLine, ...]:
    is_current_track = _is_current_track_record(record, current_info)
    watch_best = _watch_track_value(record, best_finish_times)
    watch_latest = _watch_track_value(record, latest_finish_times)
    watch_latest_delta = _watch_track_value(record, latest_finish_deltas_ms)
    failed_attempt = _has_failed_attempt(record, failed_track_attempts) and watch_best is None
    best_time = _optional_int_info(record, "track_non_agg_best_time_ms")
    worst_time = _optional_int_info(record, "track_non_agg_worst_time_ms")
    status_icon, status_color = _track_record_status(
        watch_best_ms=watch_best,
        best_time_ms=best_time,
        worst_time_ms=worst_time,
        failed_attempt=failed_attempt,
    )
    status_text = _track_record_gap_text(
        watch_best_ms=watch_best,
        best_time_ms=best_time,
        worst_time_ms=worst_time,
        failed_attempt=failed_attempt,
    )
    return (
        panel_line(
            _format_track_record_heading(record, is_current_track=is_current_track),
            "",
            status_color,
            heading=True,
            status_icon=status_icon,
            status_text="LIVE" if is_current_track and not status_text else status_text,
            label_color=PALETTE.text_accent if is_current_track else None,
            click_course_id=_record_course_id(record),
        ),
        panel_line(
            "PB",
            _format_optional_compact_time(watch_best),
            status_color if watch_best is not None else PALETTE.text_muted,
        ),
        panel_line(
            "Latest",
            _format_latest_compact_time(
                watch_latest,
                watch_best,
                latest_delta_ms=watch_latest_delta,
                failed_attempt=failed_attempt,
            ),
            _latest_time_color(
                latest_time_ms=watch_latest,
                best_time_ms=watch_best,
                latest_delta_ms=watch_latest_delta,
                failed_attempt=failed_attempt,
            ),
        ),
        panel_line(
            "WR",
            _format_record_range(best_time, worst_time),
            PALETTE.text_primary
            if best_time is not None or worst_time is not None
            else PALETTE.text_muted,
        ),
    )


def _format_track_record_heading(record: dict[str, object], *, is_current_track: bool) -> str:
    label = _format_track_record_label(record)
    return f"> {label}" if is_current_track else label


def _has_failed_attempt(
    record: dict[str, object],
    failed_track_attempts: frozenset[str],
) -> bool:
    track_key = _track_best_key(record)
    return track_key is not None and track_key in failed_track_attempts


def _record_course_id(record: dict[str, object]) -> str | None:
    course_id = record.get("track_course_id")
    if isinstance(course_id, str) and course_id:
        return course_id
    return None


def _is_current_track_record(
    record: dict[str, object],
    current_info: dict[str, object],
) -> bool:
    current_key = _track_best_key(current_info)
    record_key = _track_best_key(record)
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


def _track_record_gap_text(
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


def _format_latest_compact_time(
    latest_time_ms: int | None,
    best_time_ms: int | None,
    *,
    latest_delta_ms: int | None,
    failed_attempt: bool,
) -> str:
    if latest_time_ms is None:
        return "failed" if failed_attempt else "--"
    latest = _format_compact_race_time_ms(latest_time_ms)
    if best_time_ms is None:
        return latest

    # Delta is latest finish time minus the comparison PB. Negative is faster.
    delta_ms = latest_delta_ms if latest_delta_ms is not None else latest_time_ms - best_time_ms
    if delta_ms == 0:
        return latest
    sign = "+" if delta_ms > 0 else "-"
    return f"{latest} ({sign}{_format_compact_duration_ms(abs(delta_ms))})"


def _latest_time_color(
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


def _watch_track_value(
    info: dict[str, object],
    values: dict[str, int],
) -> int | None:
    track_key = _track_best_key(info)
    if track_key is None:
        return None
    return values.get(track_key)


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
