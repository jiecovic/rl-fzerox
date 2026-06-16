# src/rl_fzerox/ui/watch/runtime/career_mode/recording.py
from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Protocol

import numpy as np

from fzerox_emulator.arrays import Pcm16Samples, RgbFrame
from rl_fzerox.apps.recording.video import (
    FfmpegRgbWriter,
    as_rgb_frame,
    concat_mp4_recordings,
    remux_recording_to_mp4,
    resolve_ffmpeg_path,
    resolve_video_fps,
)
from rl_fzerox.core.runtime_spec.schema import WatchAppConfig
from rl_fzerox.ui.watch.runtime.career_mode.recording_hud import (
    INPUT_HUD_INFO_KEYS,
    input_hud_frame,
)


class FrameRecorder(Protocol):
    """Sink for game frames emitted by the Career Mode worker."""

    def record_frame(
        self,
        frame: RgbFrame,
        *,
        info: Mapping[str, object],
        audio_samples: Pcm16Samples = (),
    ) -> None: ...

    def record_event(self, *, info: Mapping[str, object]) -> None: ...

    def finish_segment(
        self,
        *,
        status: RecordingSegmentStatus,
        info: Mapping[str, object],
    ) -> None: ...

    def drain_notices(self) -> tuple[str, ...]: ...


class _RgbVideoWriter(Protocol):
    def __enter__(self) -> _RgbVideoWriter: ...

    def write(self, frame: RgbFrame) -> None: ...

    def write_audio(self, samples: Pcm16Samples) -> None: ...

    def close(self) -> None: ...


class _RecordingFinalizer(Protocol):
    def finalize(self, path: Path, *, summary: _SegmentSummarySnapshot | None = None) -> None: ...

    def drain_notices(self) -> tuple[str, ...]: ...

    def close(self) -> None: ...


_WriterFactory = Callable[[Path], _RgbVideoWriter]
_FinalizerFactory = Callable[[], _RecordingFinalizer]
RecordingSegmentStatus = Literal["succeeded", "failed"]


@dataclass(frozen=True, slots=True)
class _SegmentIdentity:
    key: str
    label: str
    attempt_id: str | None


@dataclass(frozen=True, slots=True)
class _FinalizerJob:
    path: Path
    future: Future[Path]
    summary: _SegmentSummarySnapshot | None = None
    delete_source: bool = False


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
    last_course_result_signature: tuple[object, ...] | None = None
    last_policy_checkpoint_signature: tuple[object, ...] | None = None

    def observe(self, info: Mapping[str, object]) -> None:
        self.frame_count += 1
        self.observe_event(info)

    def observe_event(self, info: Mapping[str, object]) -> None:
        if status := _last_finished_attempt_status(info):
            self.status = status
        self.final_info = _selected_summary_info(info)
        course_result = _course_result(info)
        if course_result is not None:
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

    def snapshot(self, *, source_path: Path, status: str | None) -> _SegmentSummarySnapshot:
        return _SegmentSummarySnapshot(
            segment_index=self.segment_index,
            label=self.label,
            attempt_id=self.attempt_id,
            status=status or self.status,
            source_path=source_path,
            started_at_utc=self.started_at_utc,
            closed_at_utc=_utc_timestamp(),
            frame_count=self.frame_count,
            course_results=tuple(dict(result) for result in self.course_results),
            policy_checkpoints=tuple(dict(checkpoint) for checkpoint in self.policy_checkpoints),
            final_info=dict(self.final_info or {}),
        )


class CareerModeFrameRecorder:
    """Record raw Career Mode game frames at native game playback speed.

    Segment lifecycle is intentionally controller-driven. The Career FSM knows
    when a cup attempt exits through success, game-over, or retry flow; this
    class only records frames/events and executes explicit finish_segment()
    commands. Keeping that boundary prevents recording from depending on
    manager DB progress timing or ad-hoc telemetry guesses.
    """

    def __init__(
        self,
        *,
        path: Path,
        native_fps: float,
        native_sample_rate: float | None = None,
        upscale_factor: int = 1,
        live_enabled: bool = True,
        live_upscale_factor: int = 1,
        session_mp4_enabled: bool = True,
        writer_factory: _WriterFactory | None = None,
        finalizer_factory: _FinalizerFactory | None = None,
    ) -> None:
        self._path = path.expanduser()
        self._fps = resolve_video_fps(native_fps=native_fps, override=None)
        self._sample_rate = _resolve_audio_sample_rate(native_sample_rate)
        self._upscale_factor = _validated_upscale_factor(upscale_factor)
        self._live_upscale_factor = _validated_upscale_factor(live_upscale_factor)
        self._writer_factory = writer_factory or self._default_writer
        self._live_path = career_live_recording_path(self._path)
        self._finalizer = (
            finalizer_factory()
            if finalizer_factory is not None
            else _Mp4RecordingFinalizer(
                session_source_path=self._path,
                live_source_path=self._live_path if live_enabled else None,
                session_mp4_enabled=session_mp4_enabled,
            )
        )
        self._live_writer = self._open_writer(self._live_path) if live_enabled else None
        self._segment_writer: _RgbVideoWriter | None = None
        self._segment_key: str | None = None
        self._segment_label: str | None = None
        self._segment_attempt_id: str | None = None
        self._segment_status: str | None = None
        self._segment_path: Path | None = None
        self._segment_summary: _SegmentSummaryBuilder | None = None
        self._segment_index = 0
        self._last_closed_finished_attempt_id: str | None = None
        self._previous_input_hud_info: dict[str, object] | None = None

    def record_frame(
        self,
        frame: RgbFrame,
        *,
        info: Mapping[str, object],
        audio_samples: Pcm16Samples = (),
    ) -> None:
        normalized_frame = as_rgb_frame(frame)
        delayed_info = self._delayed_input_hud_info(info)
        if self._live_writer is not None:
            live_frame = _recording_output_frame(
                normalized_frame,
                delayed_info,
                upscale_factor=self._live_upscale_factor,
            )
            self._live_writer.write(live_frame)
            self._live_writer.write_audio(audio_samples)
        segment = self._recording_segment(info)
        if segment is None:
            return
        segment_frame = _recording_output_frame(
            normalized_frame,
            delayed_info,
            upscale_factor=self._upscale_factor,
        )
        if segment.key != self._segment_key:
            self._start_segment(segment)
        if self._segment_writer is not None:
            if self._segment_summary is not None:
                self._segment_summary.observe(info)
            self._segment_writer.write(segment_frame)
            self._segment_writer.write_audio(audio_samples)

    def record_event(self, *, info: Mapping[str, object]) -> None:
        self._update_segment_status(info)
        if self._segment_summary is not None:
            self._segment_summary.observe_event(info)

    def finish_segment(
        self,
        *,
        status: RecordingSegmentStatus,
        info: Mapping[str, object],
    ) -> None:
        self.record_event(info=info)
        self._segment_status = status
        self._close_segment_writer()

    def _delayed_input_hud_info(self, info: Mapping[str, object]) -> Mapping[str, object]:
        if info.get("watch_recording_input_hud") is not True:
            self._previous_input_hud_info = None
            return info
        previous_info = self._previous_input_hud_info
        self._previous_input_hud_info = dict(info)
        if previous_info is None:
            return info
        delayed_info = dict(info)
        for key in INPUT_HUD_INFO_KEYS:
            if key in previous_info:
                delayed_info[key] = previous_info[key]
        return delayed_info

    def close(self) -> None:
        errors: list[Exception] = []
        try:
            self._close_segment_writer()
        except Exception as exc:  # pragma: no cover - defensive close aggregation
            errors.append(exc)
        if self._live_writer is not None:
            try:
                self._live_writer.close()
            except Exception as exc:  # pragma: no cover - defensive close aggregation
                errors.append(exc)
        try:
            self._finalizer.close()
        except Exception as exc:  # pragma: no cover - defensive close aggregation
            errors.append(exc)
        if errors:
            raise errors[0]

    def drain_notices(self) -> tuple[str, ...]:
        return self._finalizer.drain_notices()

    def _start_segment(self, segment: _SegmentIdentity) -> None:
        self._close_segment_writer()
        self._segment_index += 1
        self._segment_key = segment.key
        self._segment_label = segment.label
        self._segment_attempt_id = segment.attempt_id
        self._segment_status = None
        self._segment_path = career_segment_recording_path(
            self._path,
            segment_index=self._segment_index,
            label=segment.label,
        )
        self._segment_summary = _SegmentSummaryBuilder(
            segment_index=self._segment_index,
            label=segment.label,
            attempt_id=segment.attempt_id,
            started_at_utc=_utc_timestamp(),
        )
        self._segment_writer = self._open_writer(self._segment_path)

    def _close_segment_writer(self) -> None:
        if self._segment_writer is None:
            return
        self._segment_writer.close()
        self._segment_writer = None
        self._rename_failed_segment()
        if self._segment_path is not None:
            summary = (
                None
                if self._segment_summary is None
                else self._segment_summary.snapshot(
                    source_path=self._segment_path,
                    status=self._segment_status,
                )
            )
            self._finalizer.finalize(self._segment_path, summary=summary)
        if self._segment_status is not None:
            self._last_closed_finished_attempt_id = self._segment_attempt_id
        self._clear_segment_state()

    def _clear_segment_state(self) -> None:
        self._segment_key = None
        self._segment_label = None
        self._segment_attempt_id = None
        self._segment_status = None
        self._segment_path = None
        self._segment_summary = None

    def _rename_failed_segment(self) -> None:
        if (
            self._segment_status != "failed"
            or self._segment_path is None
            or self._segment_label is None
            or not self._segment_path.exists()
        ):
            return
        failed_path = career_segment_recording_path(
            self._path,
            segment_index=self._segment_index,
            label=self._segment_label,
            status="failed",
        )
        if failed_path != self._segment_path:
            self._segment_path.replace(failed_path)
            self._segment_path = failed_path

    def _default_writer(self, path: Path) -> FfmpegRgbWriter:
        return FfmpegRgbWriter(
            path=path,
            ffmpeg_path=resolve_ffmpeg_path(),
            fps=self._fps,
            audio_sample_rate=self._sample_rate,
        )

    def _open_writer(self, path: Path) -> _RgbVideoWriter:
        writer = self._writer_factory(path)
        return writer.__enter__()

    def _recording_segment(self, info: Mapping[str, object]) -> _SegmentIdentity | None:
        label = _segment_label(info)
        attempt_id = _attempt_id(info)
        if (
            _continuing_race_result(info)
            and self._segment_key is not None
            and self._segment_label is not None
            and not self._next_attempt_started(info)
        ):
            return _SegmentIdentity(
                key=self._segment_key,
                label=self._segment_label,
                attempt_id=self._segment_attempt_id,
            )
        if (
            self._current_segment_attempt_finished(info)
            and self._segment_key is not None
            and self._segment_label is not None
            and not self._next_attempt_started(info)
        ):
            return _SegmentIdentity(
                key=self._segment_key,
                label=self._segment_label,
                attempt_id=self._segment_attempt_id,
            )
        if (
            _post_gp_exit_frame(info)
            and attempt_id is None
            and self._segment_key is not None
            and self._segment_label is not None
        ):
            return _SegmentIdentity(
                key=self._segment_key,
                label=self._segment_label,
                attempt_id=self._segment_attempt_id,
            )
        if label is None:
            return None
        if attempt_id is None and _last_finished_attempt_status(info) is not None:
            return None
        if self._segment_key is None and attempt_id is None and _continuing_race_result(info):
            return None
        if self._segment_key is None and attempt_id is None and _post_gp_exit_frame(info):
            return None
        if (
            self._segment_key is None
            and attempt_id == self._last_closed_finished_attempt_id
            and _continuing_race_result(info)
        ):
            return None
        key = f"attempt:{attempt_id}" if attempt_id is not None else f"target:{label}"
        return _SegmentIdentity(key=key, label=label, attempt_id=attempt_id)

    def _update_segment_status(self, info: Mapping[str, object]) -> None:
        if self._segment_attempt_id is None:
            return
        if _last_finished_attempt_id(info) != self._segment_attempt_id:
            return
        status = _last_finished_attempt_status(info)
        if status is not None:
            self._segment_status = status

    def _current_segment_attempt_finished(self, info: Mapping[str, object]) -> bool:
        if self._segment_attempt_id is None:
            return False
        return (
            _last_finished_attempt_id(info) == self._segment_attempt_id
            and _last_finished_attempt_status(info) is not None
        )

    def _next_attempt_started(self, info: Mapping[str, object]) -> bool:
        attempt_id = _attempt_id(info)
        return (
            attempt_id is not None
            and self._segment_attempt_id is not None
            and attempt_id != self._segment_attempt_id
        )


class _Mp4RecordingFinalizer:
    """Remux closed live Matroska recordings to MP4 without blocking playback."""

    def __init__(
        self,
        *,
        session_source_path: Path | None = None,
        live_source_path: Path | None = None,
        session_mp4_enabled: bool = True,
    ) -> None:
        self._ffmpeg_path = resolve_ffmpeg_path()
        self._session_summary = (
            None if session_source_path is None else _SessionSummaryWriter(session_source_path)
        )
        self._session_mp4_enabled = session_mp4_enabled
        if self._session_summary is not None and live_source_path is not None:
            self._session_summary.record_live_video(live_source_path)
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="career-recording-remux",
        )
        self._jobs: list[_FinalizerJob] = []
        self._notices: list[str] = []

    def finalize(self, path: Path, *, summary: _SegmentSummarySnapshot | None = None) -> None:
        if path.suffix.lower() != ".mkv":
            if summary is not None:
                write_segment_summary_files(summary, video_path=path)
                if self._session_summary is not None:
                    self._session_summary.record_segment(summary, video_path=path)
            return
        self._jobs.append(
            _FinalizerJob(
                path=path,
                summary=summary,
                delete_source=summary is not None,
                future=self._executor.submit(
                    remux_recording_to_mp4,
                    path,
                    ffmpeg_path=self._ffmpeg_path,
                ),
            )
        )

    def drain_notices(self) -> tuple[str, ...]:
        pending: list[_FinalizerJob] = []
        notices = self._notices
        self._notices = []
        for job in self._jobs:
            if not job.future.done():
                pending.append(job)
                continue
            notices.append(_finalizer_job_notice(job, session_summary=self._session_summary))
        self._jobs = pending
        return tuple(notices)

    def close(self) -> None:
        self._executor.shutdown(wait=True)
        for job in self._jobs:
            self._notices.append(_finalizer_job_notice(job, session_summary=self._session_summary))
        self._jobs.clear()
        self._finalize_session_video()

    def _finalize_session_video(self) -> None:
        if self._session_summary is None:
            return
        if not self._session_mp4_enabled:
            self._session_summary.write()
            return
        segment_paths = self._session_summary.segment_video_paths()
        if not segment_paths:
            self._session_summary.write()
            return
        try:
            session_video_path = concat_mp4_recordings(
                segment_paths,
                ffmpeg_path=self._ffmpeg_path,
                output_path=career_session_video_path(self._session_summary.source_path),
            )
        except Exception as exc:  # pragma: no cover - defensive async error reporting
            self._notices.append(f"session MP4 assembly failed: {exc}")
            self._session_summary.write()
            return
        self._session_summary.record_session_video(session_video_path)
        self._notices.append(f"session MP4 ready: {session_video_path.name}")


def open_career_mode_recorder(
    *,
    config: WatchAppConfig,
    native_fps: float,
    native_sample_rate: float | None = None,
) -> CareerModeFrameRecorder | None:
    recording = config.watch.recording
    if not recording.enabled:
        return None
    if recording.path is None:
        raise ValueError("watch.recording.path is required when recording is enabled")
    return CareerModeFrameRecorder(
        path=recording.path,
        native_fps=native_fps,
        native_sample_rate=native_sample_rate,
        upscale_factor=recording.upscale_factor,
        live_enabled=recording.session_mp4_enabled,
        session_mp4_enabled=recording.session_mp4_enabled,
    )


def write_segment_summary_files(
    summary: _SegmentSummarySnapshot,
    *,
    video_path: Path,
) -> None:
    """Write machine-readable and human-readable sidecars for one finalized segment."""

    payload = _segment_summary_payload(summary, video_path=video_path)
    json_path = _segment_summary_path(video_path, ".json")
    markdown_path = _segment_summary_path(video_path, ".md")
    json_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(
        _segment_summary_markdown(payload),
        encoding="utf-8",
    )


def career_segment_recording_path(
    path: Path,
    *,
    segment_index: int,
    label: str,
    status: str | None = None,
) -> Path:
    suffix = path.suffix or ".mkv"
    slug = _slug(label)
    if status == "failed":
        slug = f"failed-attempt-{slug}"
    return path.with_name(f"{path.stem}.segment-{segment_index:03d}-{slug}{suffix}")


def career_session_summary_path(path: Path) -> Path:
    return path.with_name(f"{path.stem}.session.json")


def career_live_recording_path(path: Path) -> Path:
    suffix = path.suffix or ".mkv"
    return path.with_name(f"{path.stem}.live{suffix}")


def career_session_video_path(path: Path) -> Path:
    return path.with_name(f"{path.stem}.session.mp4")


def _recording_output_frame(
    frame: RgbFrame,
    info: Mapping[str, object],
    *,
    upscale_factor: int,
) -> RgbFrame:
    scaled_frame = _upscale_frame_nearest(frame, factor=upscale_factor)
    return input_hud_frame(scaled_frame, info)


def _upscale_frame_nearest(frame: RgbFrame, *, factor: int) -> RgbFrame:
    validated_factor = _validated_upscale_factor(factor)
    if validated_factor == 1:
        return frame
    repeated_rows = np.repeat(frame, validated_factor, axis=0)
    repeated_pixels = np.repeat(repeated_rows, validated_factor, axis=1)
    return as_rgb_frame(repeated_pixels)


def _validated_upscale_factor(factor: int) -> int:
    if not 1 <= factor <= 4:
        raise ValueError("recording upscale factor must be an integer from 1 to 4")
    return factor


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
                    _summary_text(course.get("engine_setting_raw_value")),
                )
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def _segment_summary_path(video_path: Path, suffix: str) -> Path:
    return video_path.with_name(f"{video_path.stem}.summary{suffix}")


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
        result[output_key] = _summary_json_value(info[key])
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


def _utc_timestamp() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


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


def _segment_label(info: Mapping[str, object]) -> str | None:
    label = info.get("career_mode_target_label")
    if not isinstance(label, str):
        return None
    stripped = label.strip()
    return stripped or None


def _attempt_id(info: Mapping[str, object]) -> str | None:
    attempt_id = info.get("career_mode_attempt_id")
    if not isinstance(attempt_id, str):
        return None
    stripped = attempt_id.strip()
    return stripped or None


def _str_info(info: Mapping[str, object], key: str) -> str | None:
    value = info.get(key)
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _last_finished_attempt_id(info: Mapping[str, object]) -> str | None:
    attempt_id = info.get("career_mode_last_finished_attempt_id")
    return attempt_id if isinstance(attempt_id, str) and attempt_id else None


def _last_finished_attempt_status(info: Mapping[str, object]) -> str | None:
    status = info.get("career_mode_last_finished_attempt_status")
    if isinstance(status, str) and status in {"succeeded", "failed"}:
        return status
    return None


def _continuing_race_result(info: Mapping[str, object]) -> bool:
    return info.get("career_mode_fsm_continuing_result") is True


def _post_gp_exit_frame(info: Mapping[str, object]) -> bool:
    return (
        info.get("career_mode_fsm_observed_screen") in _POST_GP_EXIT_SCREENS
        or info.get("game_mode") in _POST_GP_EXIT_MODES
        or info.get("game_mode_name") in _POST_GP_EXIT_MODES
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


def _finalizer_job_notice(
    job: _FinalizerJob,
    *,
    session_summary: _SessionSummaryWriter | None = None,
) -> str:
    try:
        output_path = job.future.result()
        if job.summary is not None:
            write_segment_summary_files(job.summary, video_path=output_path)
            if session_summary is not None:
                session_summary.record_segment(job.summary, video_path=output_path)
        if job.delete_source:
            job.path.unlink(missing_ok=True)
    except Exception as exc:  # pragma: no cover - defensive async error reporting
        return f"MP4 conversion failed: {exc}"
    return f"MP4 ready: {output_path.name}"


def _slug(label: str) -> str:
    parts = re.findall(r"[a-z0-9]+", label.lower())
    return "-".join(parts) if parts else "career-target"


def _resolve_audio_sample_rate(native_sample_rate: float | None) -> int | None:
    if native_sample_rate is None or native_sample_rate <= 0.0:
        return None
    return max(1, int(round(native_sample_rate)))
