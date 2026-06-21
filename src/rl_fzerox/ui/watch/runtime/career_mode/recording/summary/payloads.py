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
