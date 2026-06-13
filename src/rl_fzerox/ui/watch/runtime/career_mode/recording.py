# src/rl_fzerox/ui/watch/runtime/career_mode/recording.py
from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from fzerox_emulator.arrays import Pcm16Samples, RgbFrame
from rl_fzerox.apps.recording.video import (
    FfmpegRgbWriter,
    as_rgb_frame,
    resolve_ffmpeg_path,
    resolve_video_fps,
)
from rl_fzerox.core.runtime_spec.schema import WatchAppConfig


class FrameRecorder(Protocol):
    """Sink for game frames emitted by the Career Mode worker."""

    def record_frame(
        self,
        frame: RgbFrame,
        *,
        info: Mapping[str, object],
        audio_samples: Pcm16Samples = (),
    ) -> None: ...


class _RgbVideoWriter(Protocol):
    def __enter__(self) -> _RgbVideoWriter: ...

    def write(self, frame: RgbFrame) -> None: ...

    def write_audio(self, samples: Pcm16Samples) -> None: ...

    def close(self) -> None: ...


_WriterFactory = Callable[[Path], _RgbVideoWriter]


@dataclass(frozen=True, slots=True)
class _SegmentIdentity:
    key: str
    label: str
    attempt_id: str | None


class CareerModeFrameRecorder:
    """Record raw Career Mode game frames at native game playback speed."""

    def __init__(
        self,
        *,
        path: Path,
        native_fps: float,
        native_sample_rate: float | None = None,
        writer_factory: _WriterFactory | None = None,
    ) -> None:
        self._path = path.expanduser()
        self._fps = resolve_video_fps(native_fps=native_fps, override=None)
        self._sample_rate = _resolve_audio_sample_rate(native_sample_rate)
        self._writer_factory = writer_factory or self._default_writer
        self._full_writer = self._open_writer(self._path)
        self._segment_writer: _RgbVideoWriter | None = None
        self._segment_key: str | None = None
        self._segment_label: str | None = None
        self._segment_attempt_id: str | None = None
        self._segment_status: str | None = None
        self._segment_path: Path | None = None
        self._segment_index = 0

    def record_frame(
        self,
        frame: RgbFrame,
        *,
        info: Mapping[str, object],
        audio_samples: Pcm16Samples = (),
    ) -> None:
        normalized_frame = as_rgb_frame(frame)
        self._full_writer.write(normalized_frame)
        self._full_writer.write_audio(audio_samples)
        segment = self._recording_segment(info)
        if segment is None:
            return
        if segment.key != self._segment_key:
            self._start_segment(segment)
        self._update_segment_status(info)
        if self._segment_writer is not None:
            self._segment_writer.write(normalized_frame)
            self._segment_writer.write_audio(audio_samples)

    def close(self) -> None:
        errors: list[Exception] = []
        try:
            self._close_segment_writer()
        except Exception as exc:  # pragma: no cover - defensive close aggregation
            errors.append(exc)
        try:
            self._full_writer.close()
        except Exception as exc:  # pragma: no cover - defensive close aggregation
            errors.append(exc)
        if errors:
            raise errors[0]

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
        self._segment_writer = self._open_writer(self._segment_path)

    def _close_segment_writer(self) -> None:
        if self._segment_writer is None:
            return
        self._segment_writer.close()
        self._segment_writer = None
        self._rename_failed_segment()

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
        if (
            _continuing_race_result(info)
            and self._segment_key is not None
            and self._segment_label is not None
        ):
            return _SegmentIdentity(
                key=self._segment_key,
                label=self._segment_label,
                attempt_id=self._segment_attempt_id,
            )
        label = _segment_label(info)
        if label is None:
            return None
        attempt_id = _attempt_id(info)
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


def _slug(label: str) -> str:
    parts = re.findall(r"[a-z0-9]+", label.lower())
    return "-".join(parts) if parts else "career-target"


def _resolve_audio_sample_rate(native_sample_rate: float | None) -> int | None:
    if native_sample_rate is None or native_sample_rate <= 0.0:
        return None
    return max(1, int(round(native_sample_rate)))
