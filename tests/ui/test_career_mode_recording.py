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
        self.closed = False

    def __enter__(self) -> _FakeWriter:
        return self

    def write(self, frame: RgbFrame) -> None:
        self.frames.append(np.array(frame, copy=True))

    def close(self) -> None:
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

    recorder.record_frame(frame, info={"career_mode_target_label": "Clear Novice Jack Cup"})
    recorder.record_frame(frame, info={"career_mode_target_label": "Clear Novice Jack Cup"})
    recorder.record_frame(frame, info={"career_mode_target_label": "Clear Standard Queen Cup"})
    recorder.close()

    assert [writer.path.name for writer in writers] == [
        "career.mkv",
        "career.segment-001-clear-novice-jack-cup.mkv",
        "career.segment-002-clear-standard-queen-cup.mkv",
    ]
    assert [len(writer.frames) for writer in writers] == [3, 2, 1]
    assert all(writer.closed for writer in writers)


def test_career_segment_recording_path_sanitizes_label() -> None:
    assert (
        career_segment_recording_path(
            Path("career.mkv"),
            segment_index=7,
            label="Clear Expert Joker Cup!",
        ).name
        == "career.segment-007-clear-expert-joker-cup.mkv"
    )
