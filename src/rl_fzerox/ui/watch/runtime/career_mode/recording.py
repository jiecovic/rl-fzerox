# src/rl_fzerox/ui/watch/runtime/career_mode/recording.py
from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Protocol

from fzerox_emulator.arrays import RgbFrame
from rl_fzerox.apps.recording.video import (
    FfmpegRgbWriter,
    as_rgb_frame,
    resolve_ffmpeg_path,
    resolve_video_fps,
)
from rl_fzerox.core.runtime_spec.schema import WatchAppConfig


class FrameRecorder(Protocol):
    """Sink for game frames emitted by the Career Mode worker."""

    def record_frame(self, frame: RgbFrame, *, info: Mapping[str, object]) -> None: ...


class _RgbVideoWriter(Protocol):
    def __enter__(self) -> _RgbVideoWriter: ...

    def write(self, frame: RgbFrame) -> None: ...

    def close(self) -> None: ...


_WriterFactory = Callable[[Path], _RgbVideoWriter]


class CareerModeFrameRecorder:
    """Record raw Career Mode game frames at native game playback speed."""

    def __init__(
        self,
        *,
        path: Path,
        native_fps: float,
        writer_factory: _WriterFactory | None = None,
    ) -> None:
        self._path = path.expanduser()
        self._fps = resolve_video_fps(native_fps=native_fps, override=None)
        self._writer_factory = writer_factory or self._default_writer
        self._full_writer = self._open_writer(self._path)
        self._segment_writer: _RgbVideoWriter | None = None
        self._segment_label: str | None = None
        self._segment_index = 0

    def record_frame(self, frame: RgbFrame, *, info: Mapping[str, object]) -> None:
        normalized_frame = as_rgb_frame(frame)
        self._full_writer.write(normalized_frame)
        segment_label = _segment_label(info)
        if segment_label is None:
            return
        if segment_label != self._segment_label:
            self._start_segment(segment_label)
        if self._segment_writer is not None:
            self._segment_writer.write(normalized_frame)

    def close(self) -> None:
        errors: list[Exception] = []
        for writer in (self._segment_writer, self._full_writer):
            if writer is None:
                continue
            try:
                writer.close()
            except Exception as exc:  # pragma: no cover - defensive close aggregation
                errors.append(exc)
        self._segment_writer = None
        if errors:
            raise errors[0]

    def _start_segment(self, label: str) -> None:
        if self._segment_writer is not None:
            self._segment_writer.close()
        self._segment_index += 1
        self._segment_label = label
        self._segment_writer = self._open_writer(
            career_segment_recording_path(
                self._path,
                segment_index=self._segment_index,
                label=label,
            )
        )

    def _default_writer(self, path: Path) -> FfmpegRgbWriter:
        return FfmpegRgbWriter(
            path=path,
            ffmpeg_path=resolve_ffmpeg_path(),
            fps=self._fps,
        )

    def _open_writer(self, path: Path) -> _RgbVideoWriter:
        writer = self._writer_factory(path)
        return writer.__enter__()


def open_career_mode_recorder(
    *,
    config: WatchAppConfig,
    native_fps: float,
) -> CareerModeFrameRecorder | None:
    recording = config.watch.recording
    if not recording.enabled:
        return None
    if recording.path is None:
        raise ValueError("watch.recording.path is required when recording is enabled")
    return CareerModeFrameRecorder(path=recording.path, native_fps=native_fps)


def career_segment_recording_path(
    path: Path,
    *,
    segment_index: int,
    label: str,
) -> Path:
    suffix = path.suffix or ".mkv"
    slug = _slug(label)
    return path.with_name(f"{path.stem}.segment-{segment_index:03d}-{slug}{suffix}")


def _segment_label(info: Mapping[str, object]) -> str | None:
    label = info.get("career_mode_target_label")
    if not isinstance(label, str):
        return None
    stripped = label.strip()
    return stripped or None


def _slug(label: str) -> str:
    parts = re.findall(r"[a-z0-9]+", label.lower())
    return "-".join(parts) if parts else "career-target"
