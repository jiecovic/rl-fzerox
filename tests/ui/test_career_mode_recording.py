# tests/ui/test_career_mode_recording.py
from __future__ import annotations

from pathlib import Path

import numpy as np

from fzerox_emulator.arrays import RgbFrame
from rl_fzerox.ui.watch.runtime.career_mode.recording import (
    CareerModeFrameRecorder,
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


def test_career_recorder_writes_full_video_and_target_segments(tmp_path: Path) -> None:
    writers: list[_FakeWriter] = []

    def writer_factory(path: Path) -> _FakeWriter:
        writer = _FakeWriter(path)
        writers.append(writer)
        return writer

    frame = np.zeros((2, 3, 3), dtype=np.uint8)
    recorder = CareerModeFrameRecorder(
        path=tmp_path / "career.mkv",
        native_fps=60.0,
        writer_factory=writer_factory,
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


def test_career_recorder_keeps_segment_during_post_result_continuation(
    tmp_path: Path,
) -> None:
    writers: list[_FakeWriter] = []

    def writer_factory(path: Path) -> _FakeWriter:
        writer = _FakeWriter(path)
        writers.append(writer)
        return writer

    frame = np.zeros((2, 3, 3), dtype=np.uint8)
    recorder = CareerModeFrameRecorder(
        path=tmp_path / "career.mkv",
        native_fps=60.0,
        writer_factory=writer_factory,
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


def test_career_recorder_segments_by_attempt_and_marks_failed_attempts(
    tmp_path: Path,
) -> None:
    writers: list[_FakeWriter] = []

    def writer_factory(path: Path) -> _FakeWriter:
        writer = _FakeWriter(path)
        writers.append(writer)
        return writer

    frame = np.zeros((2, 3, 3), dtype=np.uint8)
    recorder = CareerModeFrameRecorder(
        path=tmp_path / "career.mkv",
        native_fps=60.0,
        writer_factory=writer_factory,
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
    assert [len(writer.frames) for writer in writers] == [3, 2, 1]
    assert not (tmp_path / "career.segment-001-clear-novice-jack-cup.mkv").exists()
    assert (tmp_path / "career.segment-001-failed-attempt-clear-novice-jack-cup.mkv").exists()
    assert (tmp_path / "career.segment-002-clear-novice-jack-cup.mkv").exists()


def test_career_segment_recording_path_sanitizes_label() -> None:
    assert (
        career_segment_recording_path(
            Path("career.mkv"),
            segment_index=7,
            label="Clear Expert Joker Cup!",
        ).name
        == "career.segment-007-clear-expert-joker-cup.mkv"
    )
