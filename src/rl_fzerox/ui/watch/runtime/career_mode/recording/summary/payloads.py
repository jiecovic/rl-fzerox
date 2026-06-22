# src/rl_fzerox/ui/watch/runtime/career_mode/recording/summary/payloads.py
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

from rl_fzerox.ui.watch.runtime.career_mode.recording.summary.values import (
    _format_summary_engine,
    _format_summary_time,
    _summary_text,
)

if TYPE_CHECKING:
    from rl_fzerox.ui.watch.runtime.career_mode.recording.summary.models import (
        _SegmentSummarySnapshot,
    )
    from rl_fzerox.ui.watch.runtime.career_mode.recording.summary.writer import (
        _SessionSummaryWriter,
    )


def _session_summary_payload(writer: _SessionSummaryWriter) -> dict[str, object]:
    segment_payloads = tuple(dict(segment) for segment in writer.segment_payloads)
    return {
        "kind": "career_recording_session_summary",
        "schema_version": 1,
        "session_source_path": str(writer.source_path),
        "live_mkv_path": None if writer.live_video_path is None else str(writer.live_video_path),
        "session_mp4_path": None
        if writer.session_video_path is None
        else str(writer.session_video_path),
        "segment_count": len(segment_payloads),
        "result_counts": _aggregate_segment_result_counts(segment_payloads),
        "segments": segment_payloads,
    }


def _segment_payload_sort_key(payload: Mapping[str, object]) -> int:
    value = payload.get("segment_index")
    if isinstance(value, bool) or not isinstance(value, int):
        return 0
    return value


def _segment_summary_payload(
    summary: _SegmentSummarySnapshot,
    *,
    video_path: Path,
) -> dict[str, object]:
    course_results = tuple(dict(result) for result in summary.course_results)
    policy_checkpoints = tuple(dict(checkpoint) for checkpoint in summary.policy_checkpoints)
    segment_summary = _segment_metric_summary(
        course_results,
        final_info=summary.final_info,
    )
    return {
        "kind": "career_segment_summary",
        "schema_version": 1,
        "segment_index": summary.segment_index,
        "label": summary.label,
        "attempt_id": summary.attempt_id,
        "status": summary.status,
        "started_at_utc": summary.started_at_utc,
        "closed_at_utc": summary.closed_at_utc,
        "video": {
            "mp4_path": str(video_path),
            "source_mkv_path": str(summary.source_path),
            "frame_count": summary.frame_count,
        },
        "summary": segment_summary,
        "result_counts": _segment_result_counts(course_results),
        "courses": course_results,
        "policy_checkpoints": policy_checkpoints,
        "final": summary.final_info,
    }


def _segment_summary_markdown(payload: Mapping[str, object]) -> str:
    label = payload.get("label")
    status = payload.get("status")
    video = payload.get("video")
    video_path = video.get("mp4_path") if isinstance(video, Mapping) else None
    frame_count = video.get("frame_count") if isinstance(video, Mapping) else None
    lines = [
        f"# {label if isinstance(label, str) and label else 'Career segment'}",
        "",
        f"- Status: {_summary_text(status)}",
        f"- Attempt: {_summary_text(payload.get('attempt_id'))}",
        f"- Started: {_summary_text(payload.get('started_at_utc'))}",
        f"- Closed: {_summary_text(payload.get('closed_at_utc'))}",
        f"- Video: {_summary_text(video_path)}",
        f"- Frames: {_summary_text(frame_count)}",
        "",
    ]
    lines.extend(_segment_metric_markdown_lines(payload))
    lines.extend(_policy_checkpoint_markdown_lines(payload))
    lines.extend(
        (
            "## Course results",
            "",
        )
    )
    courses = payload.get("courses")
    if not isinstance(courses, tuple | list) or not courses:
        lines.append("No course terminal results captured.")
        lines.append("")
        return "\n".join(lines)

    lines.extend(
        (
            "| Course | Result | Time | Position | KO stars | Engine |",
            "| --- | --- | --- | --- | --- | --- |",
        )
    )
    for course in courses:
        if not isinstance(course, Mapping):
            continue
        lines.append(
            "| "
            + " | ".join(
                (
                    _summary_text(
                        course.get("course_name")
                        or course.get("course_id")
                        or course.get("track_id")
                    ),
                    _summary_text(course.get("termination_reason")),
                    _format_summary_time(course.get("race_time_ms")),
                    _summary_text(course.get("position")),
                    _summary_text(course.get("ko_star_count")),
                    _format_summary_engine(course.get("engine_setting_raw_value")),
                )
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def _segment_metric_markdown_lines(payload: Mapping[str, object]) -> list[str]:
    summary = payload.get("summary")
    if not isinstance(summary, Mapping):
        return []
    lines = [
        "## Segment summary",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| Course attempts | {_summary_text(summary.get('course_attempt_count'))} |",
        f"| Finished attempts | {_summary_text(summary.get('finished_attempt_count'))} |",
        f"| Failed attempts | {_summary_text(summary.get('failed_attempt_count'))} |",
        (
            "| Unique finished courses | "
            f"{_summary_text(summary.get('unique_finished_course_count'))} |"
        ),
        (f"| Total race time | {_format_summary_time(summary.get('total_race_time_ms'))} |"),
        (
            "| Average race position | "
            f"{_format_average_position(summary.get('average_position'))} |"
        ),
        (
            "| Best / worst race position | "
            f"{_summary_text(summary.get('best_position'))} / "
            f"{_summary_text(summary.get('worst_position'))} |"
        ),
        f"| Final GP position | {_summary_text(summary.get('final_gp_position'))} |",
        f"| GP points | {_summary_text(summary.get('gp_points'))} |",
        f"| KO stars | {_summary_text(summary.get('ko_star_total'))} |",
        "",
    ]
    return lines


def _policy_checkpoint_markdown_lines(payload: Mapping[str, object]) -> list[str]:
    lines = [
        "## Policy checkpoints",
        "",
    ]
    checkpoints = payload.get("policy_checkpoints")
    if not isinstance(checkpoints, tuple | list) or not checkpoints:
        lines.extend(
            (
                "No policy checkpoint metadata captured.",
                "",
            )
        )
        return lines
    lines.extend(
        (
            "| Run | Artifact | Course | Steps | Local steps | Modified | Path |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        )
    )
    for checkpoint in checkpoints:
        if not isinstance(checkpoint, Mapping):
            continue
        lines.append(
            "| "
            + " | ".join(
                (
                    _summary_text(checkpoint.get("run_name") or checkpoint.get("run_id")),
                    _summary_text(checkpoint.get("artifact")),
                    _summary_text(checkpoint.get("course_id")),
                    _summary_text(checkpoint.get("num_timesteps")),
                    _summary_text(checkpoint.get("local_num_timesteps")),
                    _summary_text(checkpoint.get("mtime_utc")),
                    _summary_text(checkpoint.get("path")),
                )
            )
            + " |"
        )
    lines.append("")
    return lines


def _segment_result_counts(course_results: tuple[dict[str, object], ...]) -> dict[str, int]:
    counts = {
        "finished": 0,
        "retired": 0,
        "crashed": 0,
    }
    for result in course_results:
        reason = result.get("termination_reason")
        if isinstance(reason, str) and reason in counts:
            counts[reason] += 1
    counts["failed"] = counts["retired"] + counts["crashed"]
    return counts


def _segment_metric_summary(
    course_results: tuple[dict[str, object], ...],
    *,
    final_info: Mapping[str, object],
) -> dict[str, object]:
    counts = _segment_result_counts(course_results)
    race_times = tuple(_int_value(result.get("race_time_ms")) for result in course_results)
    positions = tuple(_int_value(result.get("position")) for result in course_results)
    ko_stars = tuple(_int_value(result.get("ko_star_count")) for result in course_results)
    valid_times = tuple(value for value in race_times if value is not None)
    valid_positions = tuple(value for value in positions if value is not None)
    valid_ko_stars = tuple(value for value in ko_stars if value is not None)
    unique_finished_courses = _unique_finished_course_count(course_results)
    return {
        "course_attempt_count": len(course_results),
        "finished_attempt_count": counts["finished"],
        "failed_attempt_count": counts["failed"],
        "retired_course_count": counts["retired"],
        "crashed_course_count": counts["crashed"],
        "unique_finished_course_count": unique_finished_courses,
        "total_race_time_ms": None if not valid_times else sum(valid_times),
        "average_position": None
        if not valid_positions
        else sum(valid_positions) / len(valid_positions),
        "best_position": None if not valid_positions else min(valid_positions),
        "worst_position": None if not valid_positions else max(valid_positions),
        "final_gp_position": _first_valid_gp_rank(
            final_info,
            ("career_mode_gp_final_rank", "gp_final_rank"),
        ),
        "gp_points": _first_valid_gp_points(
            final_info,
            ("career_mode_gp_points", "gp_points"),
        ),
        "ko_star_total": None if not valid_ko_stars else sum(valid_ko_stars),
    }


def _unique_finished_course_count(course_results: tuple[dict[str, object], ...]) -> int:
    course_ids: set[object] = set()
    for result in course_results:
        if result.get("termination_reason") != "finished":
            continue
        identity = _course_result_identity(result)
        if identity is not None:
            course_ids.add(identity)
    return len(course_ids)


def _course_result_identity(result: Mapping[str, object]) -> tuple[str, object] | None:
    for key in ("course_id", "track_id", "course_index"):
        value = result.get(key)
        if value is not None and not isinstance(value, bool):
            return (key, value)
    return None


def _first_valid_gp_rank(info: Mapping[str, object], keys: tuple[str, ...]) -> int | None:
    for key in keys:
        value = _int_value(info.get(key))
        if value is not None and 1 <= value <= 30:
            return value
    return None


def _first_valid_gp_points(info: Mapping[str, object], keys: tuple[str, ...]) -> int | None:
    for key in keys:
        value = _int_value(info.get(key))
        if value is not None and 0 <= value <= 999:
            return value
    return None


def _int_value(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _format_average_position(value: object) -> str:
    if not isinstance(value, int | float) or isinstance(value, bool):
        return "-"
    return f"{float(value):.2f}"


def _aggregate_segment_result_counts(
    segment_payloads: tuple[dict[str, object], ...],
) -> dict[str, int]:
    counts = {
        "finished": 0,
        "retired": 0,
        "crashed": 0,
        "failed": 0,
    }
    for payload in segment_payloads:
        result_counts = payload.get("result_counts")
        if not isinstance(result_counts, Mapping):
            continue
        for key in counts:
            value = result_counts.get(key)
            if isinstance(value, bool) or not isinstance(value, int):
                continue
            counts[key] += value
    return counts
