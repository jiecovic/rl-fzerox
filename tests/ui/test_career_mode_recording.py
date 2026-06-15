# tests/ui/test_career_mode_recording.py
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from fzerox_emulator.arrays import RgbFrame
from rl_fzerox.ui.watch.runtime.career_mode.recording import (
    CareerModeFrameRecorder,
    _Mp4RecordingFinalizer,
    _SegmentSummarySnapshot,
    career_segment_recording_path,
    career_session_summary_path,
    write_segment_summary_files,
)
from rl_fzerox.ui.watch.runtime.career_mode.recording_hud import input_hud_frame


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
        self.summaries: list[_SegmentSummarySnapshot | None] = []
        self.notices: list[str] = []
        self.closed = False

    def finalize(self, path: Path, *, summary: _SegmentSummarySnapshot | None = None) -> None:
        self.paths.append(path)
        self.summaries.append(summary)
        if summary is not None:
            write_segment_summary_files(summary, video_path=path.with_suffix(".mp4"))
        self.notices.append(f"MP4 ready: {path.with_suffix('.mp4').name}")

    def drain_notices(self) -> tuple[str, ...]:
        notices = tuple(self.notices)
        self.notices.clear()
        return notices

    def close(self) -> None:
        self.closed = True


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
        "career.segment-002-clear-standard-queen-cup.mkv",
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
        "career.live.mkv",
        "career.segment-001-clear-novice-jack-cup.mkv",
        "career.segment-002-clear-novice-queen-cup.mkv",
    ]
    assert [len(writer.frames) for writer in writers] == [3, 2, 1]
    assert [path.name for path in finalizer.paths] == [
        "career.segment-001-clear-novice-jack-cup.mkv",
        "career.segment-002-clear-novice-queen-cup.mkv",
    ]


def test_career_recorder_closes_failed_segment_after_terminal_result_frame(
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
            "career_mode_attempt_id": "attempt-a",
            "career_mode_target_label": "Clear Novice Jack Cup",
            "career_mode_fsm_continuing_result": False,
            "career_mode_fsm_terminal_result": True,
            "career_mode_last_finished_attempt_id": "attempt-a",
            "career_mode_last_finished_attempt_status": "failed",
        },
    )
    recorder.close()

    assert [writer.path.name for writer in writers] == [
        "career.live.mkv",
        "career.segment-001-clear-novice-jack-cup.mkv",
    ]
    assert [len(writer.frames) for writer in writers] == [2, 2]
    assert writers[1].closed
    assert [path.name for path in finalizer.paths] == [
        "career.segment-001-failed-attempt-clear-novice-jack-cup.mkv",
    ]


def test_career_recorder_restarts_segment_after_same_attempt_retry(
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
            "career_mode_target_label": "Clear Master Jack Cup",
        },
    )
    recorder.record_frame(
        frame,
        info={
            "career_mode_attempt_id": "attempt-a",
            "career_mode_target_label": "Clear Master Jack Cup",
            "career_mode_fsm_terminal_result": True,
            "career_mode_last_finished_attempt_id": "attempt-a",
            "career_mode_last_finished_attempt_status": "failed",
        },
    )
    recorder.record_frame(
        frame,
        info={
            "career_mode_attempt_id": "attempt-a",
            "career_mode_target_label": "Clear Master Jack Cup",
            "career_mode_fsm_continuing_result": True,
        },
    )
    recorder.record_frame(
        frame,
        info={
            "career_mode_attempt_id": "attempt-a",
            "career_mode_target_label": "Clear Master Jack Cup",
        },
    )
    recorder.close()

    assert [writer.path.name for writer in writers] == [
        "career.live.mkv",
        "career.segment-001-clear-master-jack-cup.mkv",
        "career.segment-002-clear-master-jack-cup.mkv",
    ]
    assert [len(writer.frames) for writer in writers] == [4, 2, 1]
    assert [path.name for path in finalizer.paths] == [
        "career.segment-001-failed-attempt-clear-master-jack-cup.mkv",
        "career.segment-002-clear-master-jack-cup.mkv",
    ]


def test_career_recorder_keeps_course_retire_inside_current_segment(
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
            "career_mode_target_label": "Clear Master Joker Cup",
        },
    )
    recorder.record_frame(
        frame,
        info={
            "career_mode_attempt_id": "attempt-a",
            "career_mode_target_label": "Clear Master Joker Cup",
            "career_mode_fsm_terminal_result": True,
            "termination_reason": "retired",
        },
    )
    recorder.record_frame(
        frame,
        info={
            "career_mode_attempt_id": "attempt-a",
            "career_mode_target_label": "Clear Master Joker Cup",
            "career_mode_fsm_continuing_result": True,
        },
    )
    recorder.record_frame(
        frame,
        info={
            "career_mode_attempt_id": "attempt-a",
            "career_mode_target_label": "Clear Master Joker Cup",
        },
    )
    recorder.close()

    assert [writer.path.name for writer in writers] == [
        "career.live.mkv",
        "career.segment-001-clear-master-joker-cup.mkv",
    ]
    assert [len(writer.frames) for writer in writers] == [4, 4]
    assert [path.name for path in finalizer.paths] == [
        "career.segment-001-clear-master-joker-cup.mkv",
    ]

    summary_json_path = tmp_path / "career.segment-001-clear-master-joker-cup.summary.json"
    summary = json.loads(summary_json_path.read_text(encoding="utf-8"))
    assert summary["status"] is None
    assert summary["result_counts"] == {"crashed": 0, "failed": 1, "finished": 0, "retired": 1}


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
        "career.live.mkv",
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
    ]


def test_career_recorder_starts_new_segment_after_succeeded_replay_attempt(
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
            "career_mode_target_label": "Clear Master King Cup",
        },
    )
    recorder.record_frame(
        frame,
        info={
            "career_mode_attempt_id": "attempt-a",
            "career_mode_target_label": "Clear Master King Cup",
            "career_mode_fsm_terminal_result": True,
            "career_mode_last_finished_attempt_id": "attempt-a",
            "career_mode_last_finished_attempt_status": "succeeded",
        },
    )
    recorder.record_frame(
        frame,
        info={
            "career_mode_attempt_id": "attempt-b",
            "career_mode_target_label": "Clear Master King Cup",
            "career_mode_fsm_continuing_result": True,
            "career_mode_last_finished_attempt_id": "attempt-a",
            "career_mode_last_finished_attempt_status": "succeeded",
        },
    )
    recorder.close()

    assert [writer.path.name for writer in writers] == [
        "career.live.mkv",
        "career.segment-001-clear-master-king-cup.mkv",
        "career.segment-002-clear-master-king-cup.mkv",
    ]
    assert [len(writer.frames) for writer in writers] == [3, 2, 1]
    assert [path.name for path in finalizer.paths] == [
        "career.segment-001-clear-master-king-cup.mkv",
        "career.segment-002-clear-master-king-cup.mkv",
    ]


def test_career_recorder_finalizes_succeeded_segment_on_post_gp_exit_frame(
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
            "career_mode_fsm_terminal_result": True,
            "career_mode_last_finished_attempt_id": "attempt-final",
            "career_mode_last_finished_attempt_status": "succeeded",
            "termination_reason": "finished",
            "race_time_ms": 92_345,
            "position": 1,
            "ko_star_count": 3,
            "track_course_id": "big_hand",
            "track_course_name": "Big Hand",
            "track_course_index": 23,
            "track_gp_difficulty": "master",
            "track_vehicle_name": "Blue Falcon",
            "track_engine_setting_raw_value": 80,
        },
    )

    assert [writer.path.name for writer in writers] == [
        "career.live.mkv",
        "career.segment-001-clear-master-joker-cup.mkv",
    ]
    assert not writers[1].closed
    assert finalizer.paths == []

    for mode in ("gp_end_cutscene", "skippable_credits", "unskippable_credits"):
        recorder.record_frame(
            frame,
            info={
                "career_mode_attempt_id": None,
                "career_mode_target_label": "Clear Master Joker Cup",
                "game_mode": mode,
                "career_mode_fsm_continuing_result": False,
                "career_mode_last_finished_attempt_id": "attempt-final",
                "career_mode_last_finished_attempt_status": "succeeded",
            },
        )

    assert not writers[1].closed
    assert finalizer.paths == []

    recorder.record_frame(
        frame,
        info={
            "career_mode_attempt_id": None,
            "career_mode_target_label": "Clear Master Joker Cup",
            "game_mode": "title",
            "career_mode_fsm_observed_screen": "title",
            "career_mode_fsm_continuing_result": False,
            "career_mode_last_finished_attempt_id": "attempt-final",
            "career_mode_last_finished_attempt_status": "succeeded",
        },
    )

    assert writers[1].closed
    assert len(writers[1].frames) == 6
    assert [path.name for path in finalizer.paths] == [
        "career.segment-001-clear-master-joker-cup.mkv",
    ]
    assert recorder.drain_notices() == ("MP4 ready: career.segment-001-clear-master-joker-cup.mp4",)
    summary_json_path = tmp_path / "career.segment-001-clear-master-joker-cup.summary.json"
    summary_md_path = tmp_path / "career.segment-001-clear-master-joker-cup.summary.md"
    summary = json.loads(summary_json_path.read_text(encoding="utf-8"))
    assert summary["label"] == "Clear Master Joker Cup"
    assert summary["status"] == "succeeded"
    assert summary["video"]["frame_count"] == 6
    assert summary["result_counts"] == {"crashed": 0, "failed": 0, "finished": 1, "retired": 0}
    assert summary["courses"] == [
        {
            "course_id": "big_hand",
            "course_index": 23,
            "course_name": "Big Hand",
            "difficulty": "master",
            "engine_setting_raw_value": 80,
            "ko_star_count": 3,
            "position": 1,
            "race_time_ms": 92_345,
            "termination_reason": "finished",
            "vehicle_name": "Blue Falcon",
        }
    ]
    assert "| Big Hand | finished | 1:32.345 | 1 | 3 | 80 |" in summary_md_path.read_text(
        encoding="utf-8"
    )

    recorder.close()

    assert [path.name for path in finalizer.paths] == [
        "career.segment-001-clear-master-joker-cup.mkv",
    ]
    assert recorder.drain_notices() == ()


def test_career_recorder_summary_captures_terminal_events_between_frames(
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
            "career_mode_target_label": "Clear Master Jack Cup",
        },
    )
    recorder.record_event(
        info={
            "career_mode_attempt_id": "attempt-a",
            "career_mode_target_label": "Clear Master Jack Cup",
            "termination_reason": "finished",
            "race_time_ms": 81_234,
            "position": 1,
            "ko_star_count": 2,
            "track_course_id": "mute_city",
            "track_course_name": "Mute City",
            "track_gp_difficulty": "master",
            "track_vehicle_name": "Blue Falcon",
            "track_engine_setting_raw_value": 50,
        }
    )
    recorder.record_frame(
        frame,
        info={
            "career_mode_attempt_id": "attempt-a",
            "career_mode_target_label": "Clear Master Jack Cup",
        },
    )
    recorder.record_event(
        info={
            "career_mode_attempt_id": "attempt-a",
            "career_mode_target_label": "Clear Master Jack Cup",
            "career_mode_last_finished_attempt_id": "attempt-a",
            "career_mode_last_finished_attempt_status": "succeeded",
            "termination_reason": "finished",
            "race_time_ms": 85_678,
            "position": 2,
            "ko_star_count": 4,
            "track_course_id": "silence",
            "track_course_name": "Silence",
            "track_gp_difficulty": "master",
            "track_vehicle_name": "Blue Falcon",
            "track_engine_setting_raw_value": 60,
        }
    )
    recorder.close()

    summary_json_path = tmp_path / "career.segment-001-clear-master-jack-cup.summary.json"
    summary_md_path = tmp_path / "career.segment-001-clear-master-jack-cup.summary.md"
    summary = json.loads(summary_json_path.read_text(encoding="utf-8"))
    assert summary["status"] == "succeeded"
    assert summary["video"]["frame_count"] == 2
    assert summary["result_counts"] == {"crashed": 0, "failed": 0, "finished": 2, "retired": 0}
    assert summary["courses"] == [
        {
            "course_id": "mute_city",
            "course_name": "Mute City",
            "difficulty": "master",
            "engine_setting_raw_value": 50,
            "ko_star_count": 2,
            "position": 1,
            "race_time_ms": 81_234,
            "termination_reason": "finished",
            "vehicle_name": "Blue Falcon",
        },
        {
            "course_id": "silence",
            "course_name": "Silence",
            "difficulty": "master",
            "engine_setting_raw_value": 60,
            "ko_star_count": 4,
            "position": 2,
            "race_time_ms": 85_678,
            "termination_reason": "finished",
            "vehicle_name": "Blue Falcon",
        },
    ]
    markdown = summary_md_path.read_text(encoding="utf-8")
    assert "| Mute City | finished | 1:21.234 | 1 | 2 | 50 |" in markdown
    assert "| Silence | finished | 1:25.678 | 2 | 4 | 60 |" in markdown


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


def test_career_mp4_finalizer_writes_session_summary_for_segments(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    concat_calls: list[tuple[tuple[Path, ...], str, Path]] = []

    def fake_resolve_ffmpeg_path() -> str:
        return "ffmpeg"

    def fake_remux_recording_to_mp4(
        path: Path,
        *,
        ffmpeg_path: str,
    ) -> Path:
        del ffmpeg_path
        return path.with_suffix(".mp4")

    def fake_concat_mp4_recordings(
        input_paths: tuple[Path, ...],
        *,
        ffmpeg_path: str,
        output_path: Path,
    ) -> Path:
        concat_calls.append((tuple(input_paths), ffmpeg_path, output_path))
        return output_path

    monkeypatch.setattr(
        "rl_fzerox.ui.watch.runtime.career_mode.recording.resolve_ffmpeg_path",
        fake_resolve_ffmpeg_path,
    )
    monkeypatch.setattr(
        "rl_fzerox.ui.watch.runtime.career_mode.recording.remux_recording_to_mp4",
        fake_remux_recording_to_mp4,
    )
    monkeypatch.setattr(
        "rl_fzerox.ui.watch.runtime.career_mode.recording.concat_mp4_recordings",
        fake_concat_mp4_recordings,
    )
    session_path = tmp_path / "career.mkv"
    live_path = tmp_path / "career.live.mkv"
    segment_a = tmp_path / "career.segment-001-clear-master-jack-cup.mkv"
    segment_b = tmp_path / "career.segment-002-clear-master-jack-cup.mkv"
    finalizer = _Mp4RecordingFinalizer(
        session_source_path=session_path,
        live_source_path=live_path,
    )

    finalizer.finalize(
        segment_b,
        summary=_SegmentSummarySnapshot(
            segment_index=2,
            label="Clear Master Jack Cup",
            attempt_id="attempt-b",
            status="failed",
            source_path=segment_b,
            started_at_utc="2026-06-15T10:02:00Z",
            closed_at_utc="2026-06-15T10:03:00Z",
            frame_count=20,
            course_results=(
                {
                    "course_name": "Silence",
                    "termination_reason": "crashed",
                    "race_time_ms": 12_000,
                },
            ),
            final_info={},
        ),
    )
    finalizer.finalize(
        segment_a,
        summary=_SegmentSummarySnapshot(
            segment_index=1,
            label="Clear Master Jack Cup",
            attempt_id="attempt-a",
            status="succeeded",
            source_path=segment_a,
            started_at_utc="2026-06-15T10:00:00Z",
            closed_at_utc="2026-06-15T10:01:00Z",
            frame_count=10,
            course_results=(
                {
                    "course_name": "Mute City",
                    "termination_reason": "finished",
                    "race_time_ms": 81_234,
                },
            ),
            final_info={},
        ),
    )
    finalizer.close()

    payload = json.loads(career_session_summary_path(session_path).read_text(encoding="utf-8"))
    assert payload["kind"] == "career_recording_session_summary"
    assert payload["session_source_path"] == str(session_path)
    assert payload["live_mkv_path"] == str(live_path)
    assert payload["session_mp4_path"] == str(tmp_path / "career.session.mp4")
    assert payload["segment_count"] == 2
    assert payload["result_counts"] == {"crashed": 1, "failed": 1, "finished": 1, "retired": 0}
    assert [segment["segment_index"] for segment in payload["segments"]] == [1, 2]
    assert [segment["video"]["mp4_path"] for segment in payload["segments"]] == [
        str(segment_a.with_suffix(".mp4")),
        str(segment_b.with_suffix(".mp4")),
    ]
    assert concat_calls == [
        (
            (segment_a.with_suffix(".mp4"), segment_b.with_suffix(".mp4")),
            "ffmpeg",
            tmp_path / "career.session.mp4",
        )
    ]


def test_career_mp4_finalizer_can_skip_session_mp4_for_single_target(
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
        del ffmpeg_path
        return path.with_suffix(".mp4")

    monkeypatch.setattr(
        "rl_fzerox.ui.watch.runtime.career_mode.recording.resolve_ffmpeg_path",
        fake_resolve_ffmpeg_path,
    )
    monkeypatch.setattr(
        "rl_fzerox.ui.watch.runtime.career_mode.recording.remux_recording_to_mp4",
        fake_remux_recording_to_mp4,
    )
    session_path = tmp_path / "career.mkv"
    segment_path = tmp_path / "career.segment-001-clear-master-jack-cup.mkv"
    finalizer = _Mp4RecordingFinalizer(
        session_source_path=session_path,
        session_mp4_enabled=False,
    )

    finalizer.finalize(
        segment_path,
        summary=_SegmentSummarySnapshot(
            segment_index=1,
            label="Clear Master Jack Cup",
            attempt_id="attempt-a",
            status="succeeded",
            source_path=segment_path,
            started_at_utc="2026-06-15T10:00:00Z",
            closed_at_utc="2026-06-15T10:01:00Z",
            frame_count=10,
            course_results=(),
            final_info={},
        ),
    )
    finalizer.close()

    payload = json.loads(career_session_summary_path(session_path).read_text(encoding="utf-8"))
    assert payload["session_mp4_path"] is None
    assert payload["live_mkv_path"] is None
    assert payload["segment_count"] == 1


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
