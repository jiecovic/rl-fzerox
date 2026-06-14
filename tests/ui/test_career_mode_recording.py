# tests/ui/test_career_mode_recording.py
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fzerox_emulator.arrays import RgbFrame
from rl_fzerox.ui.watch.runtime.career_mode.recording import (
    CareerModeFrameRecorder,
    _Mp4RecordingFinalizer,
    career_segment_recording_path,
)


class _FakeWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.frames: list[RgbFrame] = []
        self.audio_samples: list[list[int]] = []
        self.closed = False

    def __enter__(self) -> _FakeWriter:
        return self

    def write(self, frame: RgbFrame) -> None:
        self.frames.append(np.array(frame, copy=True))

    def write_audio(self, samples: object) -> None:
        self.audio_samples.append(np.asarray(samples, dtype=np.int16).tolist())

    def close(self) -> None:
        self.path.write_bytes(b"video")
        self.closed = True


class _FakeFinalizer:
    def __init__(self) -> None:
        self.paths: list[Path] = []
        self.notices: list[str] = []
        self.closed = False

    def finalize(self, path: Path) -> None:
        self.paths.append(path)
        self.notices.append(f"MP4 ready: {path.with_suffix('.mp4').name}")

    def drain_notices(self) -> tuple[str, ...]:
        notices = tuple(self.notices)
        self.notices.clear()
        return notices

    def close(self) -> None:
        self.closed = True


def test_career_recorder_writes_full_video_and_target_segments(tmp_path: Path) -> None:
    writers: list[_FakeWriter] = []
    finalizer = _FakeFinalizer()

    def writer_factory(path: Path) -> _FakeWriter:
        writer = _FakeWriter(path)
        writers.append(writer)
        return writer

    frame = np.zeros((2, 3, 3), dtype=np.uint8)
    recorder = CareerModeFrameRecorder(
        path=tmp_path / "career.mkv",
        native_fps=60.0,
        writer_factory=writer_factory,
        finalizer_factory=lambda: finalizer,
    )

    recorder.record_frame(
        frame,
        info={"career_mode_target_label": "Clear Novice Jack Cup"},
        audio_samples=np.array([1, -1], dtype=np.int16),
    )
    recorder.record_frame(
        frame,
        info={"career_mode_target_label": "Clear Novice Jack Cup"},
        audio_samples=np.array([2, -2], dtype=np.int16),
    )
    recorder.record_frame(
        frame,
        info={"career_mode_target_label": "Clear Standard Queen Cup"},
        audio_samples=np.array([3, -3], dtype=np.int16),
    )
    recorder.close()

    assert [writer.path.name for writer in writers] == [
        "career.mkv",
        "career.segment-001-clear-novice-jack-cup.mkv",
        "career.segment-002-clear-standard-queen-cup.mkv",
    ]
    assert [len(writer.frames) for writer in writers] == [3, 2, 1]
    assert [writer.audio_samples for writer in writers] == [
        [[1, -1], [2, -2], [3, -3]],
        [[1, -1], [2, -2]],
        [[3, -3]],
    ]
    assert all(writer.closed for writer in writers)
    assert [path.name for path in finalizer.paths] == [
        "career.segment-001-clear-novice-jack-cup.mkv",
        "career.segment-002-clear-standard-queen-cup.mkv",
        "career.mkv",
    ]
    assert finalizer.closed


def test_career_recorder_keeps_segment_during_post_result_continuation(
    tmp_path: Path,
) -> None:
    writers: list[_FakeWriter] = []
    finalizer = _FakeFinalizer()

    def writer_factory(path: Path) -> _FakeWriter:
        writer = _FakeWriter(path)
        writers.append(writer)
        return writer

    frame = np.zeros((2, 3, 3), dtype=np.uint8)
    recorder = CareerModeFrameRecorder(
        path=tmp_path / "career.mkv",
        native_fps=60.0,
        writer_factory=writer_factory,
        finalizer_factory=lambda: finalizer,
    )

    recorder.record_frame(
        frame,
        info={"career_mode_target_label": "Clear Novice Jack Cup"},
    )
    recorder.record_frame(
        frame,
        info={
            "career_mode_target_label": "Clear Novice Queen Cup",
            "career_mode_fsm_continuing_result": True,
        },
    )
    recorder.record_frame(
        frame,
        info={
            "career_mode_target_label": "Clear Novice Queen Cup",
            "career_mode_fsm_continuing_result": False,
        },
    )
    recorder.close()

    assert [writer.path.name for writer in writers] == [
        "career.mkv",
        "career.segment-001-clear-novice-jack-cup.mkv",
        "career.segment-002-clear-novice-queen-cup.mkv",
    ]
    assert [len(writer.frames) for writer in writers] == [3, 2, 1]
    assert [path.name for path in finalizer.paths] == [
        "career.segment-001-clear-novice-jack-cup.mkv",
        "career.segment-002-clear-novice-queen-cup.mkv",
        "career.mkv",
    ]


def test_career_recorder_segments_by_attempt_and_marks_failed_attempts(
    tmp_path: Path,
) -> None:
    writers: list[_FakeWriter] = []
    finalizer = _FakeFinalizer()

    def writer_factory(path: Path) -> _FakeWriter:
        writer = _FakeWriter(path)
        writers.append(writer)
        return writer

    frame = np.zeros((2, 3, 3), dtype=np.uint8)
    recorder = CareerModeFrameRecorder(
        path=tmp_path / "career.mkv",
        native_fps=60.0,
        writer_factory=writer_factory,
        finalizer_factory=lambda: finalizer,
    )

    recorder.record_frame(
        frame,
        info={
            "career_mode_attempt_id": "attempt-a",
            "career_mode_target_label": "Clear Novice Jack Cup",
        },
    )
    recorder.record_frame(
        frame,
        info={
            "career_mode_attempt_id": "attempt-b",
            "career_mode_target_label": "Clear Novice Jack Cup",
            "career_mode_fsm_continuing_result": True,
            "career_mode_last_finished_attempt_id": "attempt-a",
            "career_mode_last_finished_attempt_status": "failed",
        },
    )
    recorder.record_frame(
        frame,
        info={
            "career_mode_attempt_id": "attempt-b",
            "career_mode_target_label": "Clear Novice Jack Cup",
            "career_mode_fsm_continuing_result": False,
        },
    )
    recorder.close()

    assert [writer.path.name for writer in writers] == [
        "career.mkv",
        "career.segment-001-clear-novice-jack-cup.mkv",
        "career.segment-002-clear-novice-jack-cup.mkv",
    ]
    assert [len(writer.frames) for writer in writers] == [3, 1, 2]
    assert not (tmp_path / "career.segment-001-clear-novice-jack-cup.mkv").exists()
    assert (tmp_path / "career.segment-001-failed-attempt-clear-novice-jack-cup.mkv").exists()
    assert (tmp_path / "career.segment-002-clear-novice-jack-cup.mkv").exists()
    assert [path.name for path in finalizer.paths] == [
        "career.segment-001-failed-attempt-clear-novice-jack-cup.mkv",
        "career.segment-002-clear-novice-jack-cup.mkv",
        "career.mkv",
    ]


def test_career_recorder_finalizes_finished_segment_before_runner_exit(tmp_path: Path) -> None:
    writers: list[_FakeWriter] = []
    finalizer = _FakeFinalizer()

    def writer_factory(path: Path) -> _FakeWriter:
        writer = _FakeWriter(path)
        writers.append(writer)
        return writer

    frame = np.zeros((2, 3, 3), dtype=np.uint8)
    recorder = CareerModeFrameRecorder(
        path=tmp_path / "career.mkv",
        native_fps=60.0,
        writer_factory=writer_factory,
        finalizer_factory=lambda: finalizer,
    )

    recorder.record_frame(
        frame,
        info={
            "career_mode_attempt_id": "attempt-final",
            "career_mode_target_label": "Clear Master Joker Cup",
        },
    )
    recorder.record_frame(
        frame,
        info={
            "career_mode_attempt_id": None,
            "career_mode_target_label": "Clear Master Joker Cup",
            "career_mode_fsm_continuing_result": False,
            "career_mode_last_finished_attempt_id": "attempt-final",
            "career_mode_last_finished_attempt_status": "succeeded",
        },
    )

    assert [writer.path.name for writer in writers] == [
        "career.mkv",
        "career.segment-001-clear-master-joker-cup.mkv",
    ]
    assert writers[1].closed
    assert [path.name for path in finalizer.paths] == [
        "career.segment-001-clear-master-joker-cup.mkv"
    ]
    assert recorder.drain_notices() == (
        "MP4 ready: career.segment-001-clear-master-joker-cup.mp4",
    )

    recorder.record_frame(
        frame,
        info={
            "career_mode_attempt_id": None,
            "career_mode_target_label": "Clear Master Joker Cup",
            "career_mode_fsm_continuing_result": False,
            "career_mode_last_finished_attempt_id": "attempt-final",
            "career_mode_last_finished_attempt_status": "succeeded",
        },
    )
    recorder.close()

    assert len(writers[1].frames) == 1
    assert [path.name for path in finalizer.paths] == [
        "career.segment-001-clear-master-joker-cup.mkv",
        "career.mkv",
    ]
    assert recorder.drain_notices() == ("MP4 ready: career.mp4",)


def test_career_segment_recording_path_sanitizes_label() -> None:
    assert (
        career_segment_recording_path(
            Path("career.mkv"),
            segment_index=7,
            label="Clear Expert Joker Cup!",
        ).name
        == "career.segment-007-clear-expert-joker-cup.mkv"
    )


def test_career_mp4_finalizer_remuxes_mkv_segments(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[tuple[Path, str]] = []

    def fake_resolve_ffmpeg_path() -> str:
        return "ffmpeg"

    def fake_remux_recording_to_mp4(
        path: Path,
        *,
        ffmpeg_path: str,
    ) -> Path:
        calls.append((path, ffmpeg_path))
        return path.with_suffix(".mp4")

    monkeypatch.setattr(
        "rl_fzerox.ui.watch.runtime.career_mode.recording.resolve_ffmpeg_path",
        fake_resolve_ffmpeg_path,
    )
    monkeypatch.setattr(
        "rl_fzerox.ui.watch.runtime.career_mode.recording.remux_recording_to_mp4",
        fake_remux_recording_to_mp4,
    )
    finalizer = _Mp4RecordingFinalizer()
    mkv_path = tmp_path / "segment.mkv"

    finalizer.finalize(mkv_path)
    finalizer.finalize(tmp_path / "segment.mp4")
    finalizer.close()

    assert calls == [(mkv_path, "ffmpeg")]


def test_career_mp4_finalizer_reports_close_time_remux_failures(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fake_resolve_ffmpeg_path() -> str:
        return "ffmpeg"

    def fake_remux_recording_to_mp4(
        path: Path,
        *,
        ffmpeg_path: str,
    ) -> Path:
        del path, ffmpeg_path
        raise RuntimeError("remux failed")

    monkeypatch.setattr(
        "rl_fzerox.ui.watch.runtime.career_mode.recording.resolve_ffmpeg_path",
        fake_resolve_ffmpeg_path,
    )
    monkeypatch.setattr(
        "rl_fzerox.ui.watch.runtime.career_mode.recording.remux_recording_to_mp4",
        fake_remux_recording_to_mp4,
    )
    finalizer = _Mp4RecordingFinalizer()

    finalizer.finalize(tmp_path / "segment.mkv")
    finalizer.close()

    assert finalizer.drain_notices() == ("MP4 conversion failed: remux failed",)
