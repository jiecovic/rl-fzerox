# tests/ui/test_career_mode_recording_frames.py
"""Frame-writing and input-HUD tests for Career Mode recording.

These tests verify live/segment writer fan-out, audio forwarding, native-frame
HUD drawing, frame upscaling, and the intentional one-frame input HUD delay.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from rl_fzerox.ui.watch.runtime.career_mode.recording import CareerModeFrameRecorder
from rl_fzerox.ui.watch.runtime.career_mode.recording.hud import input_hud_frame
from tests.ui.career_mode_recording_support import _FakeFinalizer, _FakeWriter


def test_career_recorder_writes_live_video_and_target_segments(tmp_path: Path) -> None:
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
        "career.live.mkv",
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
        "career.segment-002-partial-clear-standard-queen-cup.mkv",
    ]
    assert finalizer.closed


def test_career_recorder_draws_input_hud_only_when_requested(tmp_path: Path) -> None:
    writers: list[_FakeWriter] = []
    finalizer = _FakeFinalizer()

    def writer_factory(path: Path) -> _FakeWriter:
        writer = _FakeWriter(path)
        writers.append(writer)
        return writer

    frame = np.zeros((222, 296, 3), dtype=np.uint8)
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
            "career_mode_target_label": "Clear Novice Jack Cup",
            "watch_recording_input_hud": True,
            "watch_recording_input_gas": True,
            "watch_recording_input_boost": True,
            "watch_recording_input_air_brake": False,
            "watch_recording_input_lean_left": True,
            "watch_recording_input_lean_right": False,
            "watch_recording_input_stick_x": 0.75,
            "watch_recording_input_pitch": -0.5,
        },
    )
    recorder.close()

    assert np.array_equal(writers[0].frames[0], frame)
    assert not np.array_equal(writers[0].frames[1], frame)
    assert np.array_equal(writers[1].frames[0], frame)
    assert not np.array_equal(writers[1].frames[1], frame)


def test_career_recorder_upscales_written_frames(tmp_path: Path) -> None:
    writers: list[_FakeWriter] = []
    finalizer = _FakeFinalizer()

    def writer_factory(path: Path) -> _FakeWriter:
        writer = _FakeWriter(path)
        writers.append(writer)
        return writer

    frame = np.arange(2 * 3 * 3, dtype=np.uint8).reshape((2, 3, 3))
    recorder = CareerModeFrameRecorder(
        path=tmp_path / "career.mkv",
        native_fps=60.0,
        upscale_factor=3,
        writer_factory=writer_factory,
        finalizer_factory=lambda: finalizer,
    )

    recorder.record_frame(frame, info={"career_mode_target_label": "Clear Novice Jack Cup"})
    recorder.close()

    expected = np.repeat(np.repeat(frame, 3, axis=0), 3, axis=1)
    assert np.array_equal(writers[0].frames[0], frame)
    assert np.array_equal(writers[1].frames[0], expected)


def test_career_recorder_delays_input_hud_by_one_frame(tmp_path: Path) -> None:
    writers: list[_FakeWriter] = []
    finalizer = _FakeFinalizer()

    def writer_factory(path: Path) -> _FakeWriter:
        writer = _FakeWriter(path)
        writers.append(writer)
        return writer

    frame = np.zeros((222, 296, 3), dtype=np.uint8)
    first_input = {
        "career_mode_target_label": "Clear Novice Jack Cup",
        "watch_recording_input_hud": True,
        "watch_recording_input_gas": True,
        "watch_recording_input_boost": False,
        "watch_recording_input_air_brake": False,
        "watch_recording_input_lean_left": False,
        "watch_recording_input_lean_right": False,
        "watch_recording_input_stick_x": -1.0,
        "watch_recording_input_pitch": -1.0,
    }
    second_input = {
        **first_input,
        "watch_recording_input_gas": False,
        "watch_recording_input_boost": True,
        "watch_recording_input_stick_x": 1.0,
        "watch_recording_input_pitch": 1.0,
    }
    recorder = CareerModeFrameRecorder(
        path=tmp_path / "career.mkv",
        native_fps=60.0,
        writer_factory=writer_factory,
        finalizer_factory=lambda: finalizer,
    )

    recorder.record_frame(frame, info=first_input)
    recorder.record_frame(frame, info=second_input)
    recorder.close()

    assert np.array_equal(writers[0].frames[0], input_hud_frame(frame, first_input))
    assert np.array_equal(writers[0].frames[1], input_hud_frame(frame, first_input))
