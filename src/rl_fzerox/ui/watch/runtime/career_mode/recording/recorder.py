# src/rl_fzerox/ui/watch/runtime/career_mode/recording/recorder.py
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

import numpy as np

from fzerox_emulator.arrays import Pcm16Samples, RgbFrame
from rl_fzerox.apps.recording.video import (
    FfmpegRgbWriter,
    as_rgb_frame,
    resolve_ffmpeg_path,
    resolve_video_fps,
)
from rl_fzerox.core.runtime_spec.schema import WatchAppConfig
from rl_fzerox.ui.watch.runtime.career_mode.recording.finalizer import (
    _Mp4RecordingFinalizer,
)
from rl_fzerox.ui.watch.runtime.career_mode.recording.hud import (
    INPUT_HUD_INFO_KEYS,
    input_hud_frame,
)
from rl_fzerox.ui.watch.runtime.career_mode.recording.paths import (
    career_live_recording_path,
    career_segment_recording_path,
)
from rl_fzerox.ui.watch.runtime.career_mode.recording.summary import (
    _SegmentSummaryBuilder,
    _SegmentSummarySnapshot,
    attempt_id,
    continuing_race_result,
    last_finished_attempt_id,
    last_finished_attempt_status,
    post_gp_exit_frame,
    segment_label,
    utc_timestamp,
)

RecordingSegmentStatus = Literal["succeeded", "failed"]


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


@dataclass(frozen=True, slots=True)
class _SegmentIdentity:
    key: str
    label: str
    attempt_id: str | None


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
        keep_failed_segments: bool = True,
        writer_factory: _WriterFactory | None = None,
        finalizer_factory: _FinalizerFactory | None = None,
    ) -> None:
        self._path = path.expanduser()
        self._fps = resolve_video_fps(native_fps=native_fps, override=None)
        self._sample_rate = _resolve_audio_sample_rate(native_sample_rate)
        self._upscale_factor = _validated_upscale_factor(upscale_factor)
        self._live_upscale_factor = _validated_upscale_factor(live_upscale_factor)
        self._keep_failed_segments = keep_failed_segments
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
            started_at_utc=utc_timestamp(),
        )
        self._segment_writer = self._open_writer(self._segment_path)

    def _close_segment_writer(self) -> None:
        if self._segment_writer is None:
            return
        self._segment_writer.close()
        self._segment_writer = None
        if self._discard_failed_segment():
            if self._segment_status is not None:
                self._last_closed_finished_attempt_id = self._segment_attempt_id
            self._clear_segment_state()
            return
        self._rename_failed_segment()
        if self._segment_path is not None and self._segment_path.exists():
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

    def _discard_failed_segment(self) -> bool:
        if self._keep_failed_segments or self._segment_status != "failed":
            return False
        if self._segment_path is not None:
            self._segment_path.unlink(missing_ok=True)
        return True

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
        label = segment_label(info)
        active_attempt_id = attempt_id(info)
        if (
            continuing_race_result(info)
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
            post_gp_exit_frame(info)
            and active_attempt_id is None
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
        if active_attempt_id is None and last_finished_attempt_status(info) is not None:
            return None
        if self._segment_key is None and active_attempt_id is None and continuing_race_result(info):
            return None
        if self._segment_key is None and active_attempt_id is None and post_gp_exit_frame(info):
            return None
        if (
            self._segment_key is None
            and active_attempt_id == self._last_closed_finished_attempt_id
            and continuing_race_result(info)
        ):
            return None
        key = f"attempt:{active_attempt_id}" if active_attempt_id is not None else f"target:{label}"
        return _SegmentIdentity(key=key, label=label, attempt_id=active_attempt_id)

    def _update_segment_status(self, info: Mapping[str, object]) -> None:
        if self._segment_attempt_id is None:
            return
        if last_finished_attempt_id(info) != self._segment_attempt_id:
            return
        status = last_finished_attempt_status(info)
        if status is not None:
            self._segment_status = status

    def _current_segment_attempt_finished(self, info: Mapping[str, object]) -> bool:
        if self._segment_attempt_id is None:
            return False
        return (
            last_finished_attempt_id(info) == self._segment_attempt_id
            and last_finished_attempt_status(info) is not None
        )

    def _next_attempt_started(self, info: Mapping[str, object]) -> bool:
        active_attempt_id = attempt_id(info)
        return (
            active_attempt_id is not None
            and self._segment_attempt_id is not None
            and active_attempt_id != self._segment_attempt_id
        )


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
        keep_failed_segments=recording.keep_failed_segments,
    )


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


def _resolve_audio_sample_rate(native_sample_rate: float | None) -> int | None:
    if native_sample_rate is None or native_sample_rate <= 0.0:
        return None
    return max(1, int(round(native_sample_rate)))
