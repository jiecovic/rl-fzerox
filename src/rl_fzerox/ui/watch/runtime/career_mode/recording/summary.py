# src/rl_fzerox/ui/watch/runtime/career_mode/recording/summary.py
from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from rl_fzerox.core.career_mode.navigation import BUILT_IN_COURSES_BY_INDEX
from rl_fzerox.core.runtime_spec.vehicle_catalog import engine_setting_display_name_for_raw
from rl_fzerox.ui.watch.runtime.career_mode.recording.paths import (
    career_session_summary_path,
    segment_summary_path,
)


@dataclass(slots=True)
class _SessionSummaryWriter:
    source_path: Path
    live_video_path: Path | None = None
    session_video_path: Path | None = None
    segment_payloads: list[dict[str, object]] = field(default_factory=list)

    def record_live_video(self, video_path: Path) -> None:
        self.live_video_path = video_path
        self.write()

    def record_session_video(self, video_path: Path) -> None:
        self.session_video_path = video_path
        self.write()

    def record_segment(self, summary: _SegmentSummarySnapshot, *, video_path: Path) -> None:
        payload = _segment_summary_payload(summary, video_path=video_path)
        self.segment_payloads = [
            segment
            for segment in self.segment_payloads
            if segment.get("segment_index") != summary.segment_index
        ]
        self.segment_payloads.append(payload)
        self.segment_payloads.sort(key=_segment_payload_sort_key)
        self.write()

    def segment_video_paths(self) -> tuple[Path, ...]:
        paths: list[Path] = []
        for payload in self.segment_payloads:
            video = payload.get("video")
            if not isinstance(video, Mapping):
                continue
            path = video.get("mp4_path")
            if isinstance(path, str) and path:
                paths.append(Path(path))
        return tuple(paths)

    def write(self) -> None:
        career_session_summary_path(self.source_path).write_text(
            json.dumps(_session_summary_payload(self), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


@dataclass(frozen=True, slots=True)
class _SegmentSummarySnapshot:
    segment_index: int
    label: str
    attempt_id: str | None
    status: str | None
    source_path: Path
    started_at_utc: str
    closed_at_utc: str
    frame_count: int
    course_results: tuple[dict[str, object], ...]
    final_info: dict[str, object]
    policy_checkpoints: tuple[dict[str, object], ...] = ()


@dataclass(slots=True)
class _SegmentSummaryBuilder:
    segment_index: int
    label: str
    attempt_id: str | None
    started_at_utc: str
    frame_count: int = 0
    status: str | None = None
    final_info: dict[str, object] | None = None
    course_results: list[dict[str, object]] = field(default_factory=list)
    policy_checkpoints: list[dict[str, object]] = field(default_factory=list)
    observed_course_engine_raw_values: dict[tuple[str, object], int] = field(default_factory=dict)
    last_course_result_signature: tuple[object, ...] | None = None
    last_policy_checkpoint_signature: tuple[object, ...] | None = None

    def observe(self, info: Mapping[str, object]) -> None:
        self.frame_count += 1
        self.observe_event(info)

    def observe_event(self, info: Mapping[str, object]) -> None:
        if status := last_finished_attempt_status(info):
            self.status = status
        self.final_info = _selected_summary_info(info)
        if engine_observation := _course_engine_observation(info):
            identity, engine_raw = engine_observation
            self.observed_course_engine_raw_values[identity] = engine_raw
        course_result = _course_result(info)
        if course_result is not None:
            self._apply_observed_course_engine(course_result)
            existing_index = _existing_course_result_index(self.course_results, course_result)
            if existing_index is not None:
                self.course_results[existing_index] = _merge_missing_course_result_fields(
                    self.course_results[existing_index],
                    course_result,
                )
                self.last_course_result_signature = _course_result_signature(course_result)
            else:
                signature = _course_result_signature(course_result)
                if signature != self.last_course_result_signature:
                    self.course_results.append(course_result)
                    self.last_course_result_signature = signature
        policy_checkpoint = _policy_checkpoint_summary(info)
        if policy_checkpoint is None:
            return
        signature = _policy_checkpoint_signature(policy_checkpoint)
        if signature == self.last_policy_checkpoint_signature:
            return
        self.policy_checkpoints.append(policy_checkpoint)
        self.last_policy_checkpoint_signature = signature

    def _apply_observed_course_engine(self, course_result: dict[str, object]) -> None:
        identity = _course_result_identity(course_result)
        if identity is None:
            return
        engine_raw = self.observed_course_engine_raw_values.get(identity)
        if engine_raw is None:
            return
        course_result["engine_setting_raw_value"] = engine_raw

    def snapshot(self, *, source_path: Path, status: str | None) -> _SegmentSummarySnapshot:
        return _SegmentSummarySnapshot(
            segment_index=self.segment_index,
            label=self.label,
            attempt_id=self.attempt_id,
            status=status or self.status,
            source_path=source_path,
            started_at_utc=self.started_at_utc,
            closed_at_utc=utc_timestamp(),
            frame_count=self.frame_count,
            course_results=tuple(dict(result) for result in self.course_results),
            policy_checkpoints=tuple(dict(checkpoint) for checkpoint in self.policy_checkpoints),
            final_info=dict(self.final_info or {}),
        )


def write_segment_summary_files(
    summary: _SegmentSummarySnapshot,
    *,
    video_path: Path,
) -> None:
    """Write machine-readable and human-readable sidecars for one finalized segment."""

    payload = _segment_summary_payload(summary, video_path=video_path)
    json_path = segment_summary_path(video_path, ".json")
    markdown_path = segment_summary_path(video_path, ".md")
    json_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(
        _segment_summary_markdown(payload),
        encoding="utf-8",
    )


def segment_label(info: Mapping[str, object]) -> str | None:
    label = info.get("career_mode_target_label")
    if not isinstance(label, str):
        return None
    stripped = label.strip()
    return stripped or None


def attempt_id(info: Mapping[str, object]) -> str | None:
    value = info.get("career_mode_attempt_id")
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def last_finished_attempt_id(info: Mapping[str, object]) -> str | None:
    attempt = info.get("career_mode_last_finished_attempt_id")
    return attempt if isinstance(attempt, str) and attempt else None


def last_finished_attempt_status(info: Mapping[str, object]) -> str | None:
    status = info.get("career_mode_last_finished_attempt_status")
    if isinstance(status, str) and status in {"succeeded", "failed"}:
        return status
    return None


def continuing_race_result(info: Mapping[str, object]) -> bool:
    return info.get("career_mode_fsm_continuing_result") is True


def post_gp_exit_frame(info: Mapping[str, object]) -> bool:
    return (
        info.get("career_mode_fsm_observed_screen") in _POST_GP_EXIT_SCREENS
        or info.get("game_mode") in _POST_GP_EXIT_MODES
        or info.get("game_mode_name") in _POST_GP_EXIT_MODES
    )


def utc_timestamp() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


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


def _selected_summary_info(info: Mapping[str, object]) -> dict[str, object]:
    selected: dict[str, object] = {}
    for key in _SUMMARY_INFO_FIELDS:
        if key not in info:
            continue
        selected[key] = _summary_json_value(info[key])
    return selected


def _course_result(info: Mapping[str, object]) -> dict[str, object] | None:
    reason = _str_info(info, "termination_reason")
    if reason not in {"finished", "retired", "crashed"}:
        return None
    result: dict[str, object] = {
        "termination_reason": reason,
    }
    for key, output_key in (
        ("track_id", "track_id"),
        ("track_course_id", "course_id"),
        ("track_course_name", "course_name"),
        ("track_course_index", "course_index"),
        ("track_gp_difficulty", "difficulty"),
        ("race_time_ms", "race_time_ms"),
        ("position", "position"),
        ("ko_star_count", "ko_star_count"),
        ("race_laps_completed", "laps_completed"),
        ("total_lap_count", "total_laps"),
        ("track_vehicle_name", "vehicle_name"),
        ("track_engine_setting_raw_value", "engine_setting_raw_value"),
    ):
        if key not in info:
            continue
        value = _summary_json_value(info[key])
        if value is not None:
            result[output_key] = value
    _add_course_result_fallbacks(result, info)
    if reason == "finished" and _course_result_identity(result) is None:
        return None
    return result


def _course_result_signature(result: Mapping[str, object]) -> tuple[object, ...]:
    return tuple(
        result.get(key)
        for key in (
            "track_id",
            "course_id",
            "course_index",
            "termination_reason",
            "race_time_ms",
            "position",
        )
    )


def _course_result_identity(result: Mapping[str, object]) -> tuple[str, object] | None:
    for key in ("course_id", "track_id", "course_index"):
        value = result.get(key)
        if value is not None:
            return (key, value)
    return None


def _course_engine_observation(
    info: Mapping[str, object],
) -> tuple[tuple[str, object], int] | None:
    if info.get("game_mode") != "gp_race" and info.get("game_mode_name") != "gp_race":
        return None
    if _str_info(info, "termination_reason") in {"finished", "retired", "crashed"}:
        return None
    identity = _course_identity_from_info(info)
    if identity is None:
        return None
    engine_raw = _int_mapping(info, "engine_setting_raw_value_ram")
    if engine_raw is None:
        engine_raw = _int_mapping(info, "track_engine_setting_raw_value")
    if engine_raw is None:
        engine_raw = _int_mapping(info, "engine_setting_raw_value")
    if engine_raw is None:
        return None
    return identity, engine_raw


def _course_identity_from_info(info: Mapping[str, object]) -> tuple[str, object] | None:
    for key in ("track_course_id", "course_id", "track_id", "career_mode_policy_course_id"):
        value = info.get(key)
        if value is None:
            continue
        if key in {"track_course_id", "career_mode_policy_course_id"}:
            return ("course_id", value)
        return (key, value)
    for key in ("track_course_index", "course_index"):
        value = info.get(key)
        if value is not None:
            if isinstance(value, int) and not isinstance(value, bool):
                course = BUILT_IN_COURSES_BY_INDEX.get(value)
                if course is not None:
                    return ("course_id", course.id)
            return ("course_index", value)
    return None


def _existing_course_result_index(
    results: list[dict[str, object]],
    candidate: Mapping[str, object],
) -> int | None:
    for index, result in enumerate(results):
        if _course_results_match(result, candidate):
            return index
    return None


def _course_results_match(
    current: Mapping[str, object],
    candidate: Mapping[str, object],
) -> bool:
    current_identity = _course_result_identity(current)
    candidate_identity = _course_result_identity(candidate)
    if (
        current_identity is not None
        and candidate_identity is not None
        and current_identity == candidate_identity
    ):
        return True

    # Live Career Mode can observe the same terminal result twice: once before
    # course metadata has been enriched and once after. Match that ordering on
    # the immutable terminal fields so the later event fills the missing course
    # name/index/engine instead of appending a duplicate row.
    if current_identity is not None and candidate_identity is not None:
        return False
    current_fingerprint = _course_terminal_fingerprint(current)
    candidate_fingerprint = _course_terminal_fingerprint(candidate)
    return (
        current_fingerprint is not None
        and candidate_fingerprint is not None
        and current_fingerprint == candidate_fingerprint
    )


def _merge_missing_course_result_fields(
    current: Mapping[str, object],
    candidate: Mapping[str, object],
) -> dict[str, object]:
    merged = dict(current)
    for key, value in candidate.items():
        if key not in merged or merged[key] is None:
            merged[key] = value
    return merged


def _add_course_result_fallbacks(
    result: dict[str, object],
    info: Mapping[str, object],
) -> None:
    course_index = _int_mapping(result, "course_index")
    if course_index is None:
        course_index = _int_mapping(info, "course_index")
    if course_index is not None:
        result.setdefault("course_index", course_index)
        course = BUILT_IN_COURSES_BY_INDEX.get(course_index)
        if course is not None:
            result.setdefault("track_id", course.id)
            result.setdefault("course_id", course.id)
            result.setdefault("course_name", course.display_name)
            difficulty = _summary_json_value(info.get("difficulty"))
            if difficulty is not None:
                result.setdefault("difficulty", difficulty)


def _course_terminal_fingerprint(result: Mapping[str, object]) -> tuple[object, ...] | None:
    race_time_ms = result.get("race_time_ms")
    if race_time_ms is None:
        return None
    return (
        result.get("termination_reason"),
        race_time_ms,
        result.get("position"),
        result.get("ko_star_count"),
        result.get("laps_completed"),
        result.get("total_laps"),
    )


def _policy_checkpoint_summary(info: Mapping[str, object]) -> dict[str, object] | None:
    path = _str_info(info, "career_mode_policy_checkpoint_path")
    if path is None:
        return None
    summary: dict[str, object] = {"path": path}
    for key, output_key in (
        ("career_mode_policy_run_id", "run_id"),
        ("career_mode_policy_run_name", "run_name"),
        ("career_mode_policy_artifact", "artifact"),
        ("career_mode_policy_course_id", "course_id"),
        ("career_mode_policy_checkpoint_num_timesteps", "num_timesteps"),
        ("career_mode_policy_checkpoint_local_num_timesteps", "local_num_timesteps"),
        ("career_mode_policy_checkpoint_mtime_utc", "mtime_utc"),
        ("career_mode_policy_checkpoint_mtime_ns", "mtime_ns"),
        ("career_mode_policy_checkpoint_stage", "stage"),
        ("career_mode_policy_checkpoint_stage_index", "stage_index"),
    ):
        if key not in info:
            continue
        summary[output_key] = _summary_json_value(info[key])
    return summary


def _policy_checkpoint_signature(checkpoint: Mapping[str, object]) -> tuple[object, ...]:
    return tuple(
        checkpoint.get(key)
        for key in (
            "run_id",
            "artifact",
            "course_id",
            "path",
            "mtime_ns",
            "num_timesteps",
            "local_num_timesteps",
        )
    )


def _summary_json_value(value: object) -> object:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _summary_text(value: object) -> str:
    if value is None:
        return "-"
    return str(value).replace("|", "\\|")


def _format_summary_time(value: object) -> str:
    if isinstance(value, bool) or not isinstance(value, int):
        return "-"
    minutes, remainder = divmod(max(0, value), 60_000)
    seconds, millis = divmod(remainder, 1_000)
    return f"{minutes}:{seconds:02d}.{millis:03d}"


def _format_summary_engine(value: object) -> str:
    if isinstance(value, bool) or not isinstance(value, int):
        return "-"
    try:
        return engine_setting_display_name_for_raw(value)
    except ValueError:
        return _summary_text(value)


def _str_info(info: Mapping[str, object], key: str) -> str | None:
    value = info.get(key)
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _int_mapping(info: Mapping[str, object], key: str) -> int | None:
    value = info.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


_SUMMARY_INFO_FIELDS = (
    "career_mode_target_label",
    "career_mode_attempt_id",
    "career_mode_last_finished_attempt_id",
    "career_mode_last_finished_attempt_status",
    "career_mode_last_finished_attempt_failure_reason",
    "career_mode_fsm_observed_screen",
    "career_mode_fsm_terminal_reason",
    "career_mode_policy_artifact",
    "career_mode_policy_checkpoint_local_num_timesteps",
    "career_mode_policy_checkpoint_mtime_ns",
    "career_mode_policy_checkpoint_mtime_utc",
    "career_mode_policy_checkpoint_num_timesteps",
    "career_mode_policy_checkpoint_path",
    "career_mode_policy_checkpoint_stage",
    "career_mode_policy_checkpoint_stage_index",
    "career_mode_policy_course_id",
    "career_mode_policy_run_id",
    "career_mode_policy_run_name",
    "game_mode",
    "game_mode_name",
    "track_id",
    "track_course_id",
    "track_course_name",
    "track_course_index",
    "track_gp_difficulty",
    "track_vehicle_name",
    "track_engine_setting_raw_value",
    "termination_reason",
    "race_time_ms",
    "position",
    "race_laps_completed",
    "total_lap_count",
)

_POST_GP_EXIT_SCREENS = frozenset(
    {
        "title",
        "main_menu_gp",
        "main_menu_other",
        "course_select",
    }
)

_POST_GP_EXIT_MODES = frozenset(
    {
        "title",
        "main_menu",
        "course_select",
        "game_over",
    }
)
